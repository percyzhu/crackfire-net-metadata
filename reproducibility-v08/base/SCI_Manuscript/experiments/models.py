"""Independent PyTorch reference model and information-controlled baselines.

The legacy GNN keeps the original checkpoint parameter names and equations.
It replaces only PyG's sum message aggregation and mean graph pooling with
PyTorch index_add. Direct numerical comparison against installed PyG remains
a release check; this module does not modify the archived implementation.
"""
import math
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.LayerNorm(dims[i + 1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mean_pool(x, batch, n_graphs):
    sums = x.new_zeros((n_graphs, x.shape[-1])).index_add(0, batch, x)
    counts = torch.bincount(batch, minlength=n_graphs).clamp_min(1)
    return sums / counts[:, None]


class CrackProcessorLayer(nn.Module):
    def __init__(self, hidden_dim, mlp_layers=2, isolated_node_policy="always_update"):
        super().__init__()
        self.isolated_node_policy = isolated_node_policy
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0 and self.isolated_node_policy == "legacy_batch_skip":
            return x, edge_attr
        src, dst = edge_index
        edge_attr = edge_attr + self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], -1))
        aggregate = torch.zeros_like(x).index_add(0, dst, edge_attr)
        return x + self.node_mlp(torch.cat([x, aggregate], -1)), edge_attr


class StructureHead(nn.Module):
    def __init__(self, crack_in_dim=11, edge_in_dim=5, global_in_dim=3,
                 hidden_dim=64, num_mp_steps=4, mlp_layers=2,
                 isolated_node_policy="always_update"):
        super().__init__()
        self.crack_encoder = MLP(crack_in_dim, hidden_dim, hidden_dim, mlp_layers)
        self.global_encoder = MLP(global_in_dim, hidden_dim, hidden_dim, mlp_layers)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, mlp_layers)
        self.processors = nn.ModuleList([
            CrackProcessorLayer(hidden_dim, mlp_layers, isolated_node_policy)
            for _ in range(num_mp_steps)])
        self.global_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(self, x, edge_index, edge_attr, x_global, batch):
        x = self.crack_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        g = self.global_encoder(x_global)
        for layer in self.processors:
            x, edge_attr = layer(x, edge_index, edge_attr)
        pooled = mean_pool(x, batch, x_global.shape[0])
        return g + self.global_update(torch.cat([g, pooled], -1))


class TemporalHead(nn.Module):
    def __init__(self, time_in_dim=4, hidden_dim=64, temporal_type="lstm",
                 mlp_layers=2, conv_kernel=5):
        super().__init__()
        self.temporal_type = temporal_type
        self.step_mlp = MLP(time_in_dim, hidden_dim, hidden_dim, mlp_layers)
        if temporal_type == "lstm":
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
            self.ln = nn.LayerNorm(hidden_dim)
        elif temporal_type == "conv1d":
            self.causal_pad = conv_kernel - 1
            self.conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, conv_kernel),
                                      nn.ReLU(), nn.LayerNorm(hidden_dim))
        elif temporal_type != "mlp_only":
            raise ValueError(temporal_type)

    def forward(self, time_features):
        h = self.step_mlp(time_features)
        if self.temporal_type == "lstm":
            return h + self.ln(self.lstm(h)[0])
        if self.temporal_type == "conv1d":
            z = nn.functional.pad(h.transpose(1, 2), (self.causal_pad, 0))
            return h + self.conv[2](self.conv[1](self.conv[0](z)).transpose(1, 2))
        return h


class BaselineStructure(nn.Module):
    def __init__(self, mode, hidden_dim=64, mlp_layers=2):
        super().__init__()
        self.mode, self.hidden_dim = mode, hidden_dim
        if mode in ("global_stats", "deepsets"):
            self.global_encoder = MLP(3, hidden_dim, hidden_dim, mlp_layers)
        if mode == "deepsets":
            self.crack_encoder = MLP(11, hidden_dim, hidden_dim, mlp_layers)
            self.global_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(self, x, edge_index, edge_attr, x_global, batch):
        if self.mode == "fire_only":
            return x_global.new_zeros((x_global.shape[0], self.hidden_dim))
        g = self.global_encoder(x_global)
        if self.mode == "global_stats":
            return g
        pooled = mean_pool(self.crack_encoder(x), batch, x_global.shape[0])
        return g + self.global_update(torch.cat([g, pooled], -1))


class Surrogate(nn.Module):
    def __init__(self, mode="gnn", hidden_dim=64, num_mp_steps=4,
                 temporal_type="lstm", mlp_layers=2, crack_in_dim=11,
                 edge_in_dim=5, global_in_dim=3, time_in_dim=4,
                 isolated_node_policy="always_update"):
        super().__init__()
        if isolated_node_policy not in ("always_update", "legacy_batch_skip"):
            raise ValueError(isolated_node_policy)
        self.mode = mode
        if mode in ("gnn", "gnn_zero_edge_features"):
            self.structure_head = StructureHead(crack_in_dim, edge_in_dim, global_in_dim,
                                                hidden_dim, num_mp_steps, mlp_layers,
                                                isolated_node_policy)
        else:
            if mode not in ("fire_only", "global_stats", "deepsets"):
                raise ValueError(mode)
            self.structure_head = BaselineStructure(mode, hidden_dim, mlp_layers)
        self.temporal_head = TemporalHead(time_in_dim, hidden_dim, temporal_type, mlp_layers)
        self.decoder = nn.Sequential(MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers),
                                     nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr, x_global, batch, time_features):
        n_graphs = x_global.shape[0]
        time_features = time_features.reshape(n_graphs, -1, time_features.shape[-1])
        if self.mode == "gnn_zero_edge_features":
            edge_attr = torch.zeros_like(edge_attr)
        structural = self.structure_head(x, edge_index, edge_attr, x_global, batch)
        temporal = self.temporal_head(time_features)
        structural = structural[:, None].expand(-1, temporal.shape[1], -1)
        return self.decoder(torch.cat([structural, temporal], -1))


def from_checkpoint_config(cfg):
    mc = cfg["model"]
    fields = ("hidden_dim", "num_mp_steps", "mlp_layers", "crack_in_dim", "edge_in_dim",
              "global_in_dim", "time_in_dim")
    kwargs = {k: mc[k] for k in fields if k in mc}
    temporal = mc.get("temporal_type", "conv1d")
    if temporal == "conv1d" and not mc.get("use_conv", True):
        temporal = "mlp_only"
    return Surrogate(mode="gnn", temporal_type=temporal,
                     isolated_node_policy="legacy_batch_skip", **kwargs)
