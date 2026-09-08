"""
CrackFireNet: Dual-head GNN for fire resistance time-series prediction.

Architecture:
    Head 1 (Structure):  Crack graph -> GNN -> structural embedding [B, D]
    Head 2 (Temporal):   (t, T_iso834) per step -> MLP -> temporal embedding [B, T, D]
    Fusion:              concat(structural.expand(T), temporal) -> [B, T, 2D]
    Decoder:             MLP -> area_ratio [B, T, 1]

Input:  Crack parameters graph + time series (t, T_iso)
Output: Time series of 300C charring area fraction (0~1) at each time step
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


# ============================================================
# Building blocks
# ============================================================

class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm and ReLU."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CrackProcessorLayer(MessagePassing):
    """Single message passing round with residual connections."""

    def __init__(self, hidden_dim, mlp_layers=2):
        super().__init__(aggr="add")
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            # No edges: skip message passing, return unchanged
            return x, edge_attr

        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_mlp(edge_input)

        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        node_input = torch.cat([x, agg], dim=-1)
        x = x + self.node_mlp(node_input)
        return x, edge_attr

    def message(self, edge_attr):
        return edge_attr


# ============================================================
# Head 1: Structure (Crack Graph)
# ============================================================

class StructureHead(nn.Module):
    """Crack graph encoder: cracks -> GNN -> structural embedding.

    Args:
        crack_in_dim:  Crack node feature dim (default 11)
        edge_in_dim:   Edge feature dim (default 5)
        global_in_dim: Global node feature dim (default 3)
        hidden_dim:    Hidden dimension
        num_mp_steps:  Message passing rounds
        mlp_layers:    Layers per MLP
    """

    def __init__(self, crack_in_dim=11, edge_in_dim=5, global_in_dim=3,
                 hidden_dim=64, num_mp_steps=4, mlp_layers=2):
        super().__init__()
        self.crack_encoder = MLP(crack_in_dim, hidden_dim, hidden_dim, mlp_layers)
        self.global_encoder = MLP(global_in_dim, hidden_dim, hidden_dim, mlp_layers)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, mlp_layers)

        self.processors = nn.ModuleList(
            [CrackProcessorLayer(hidden_dim, mlp_layers) for _ in range(num_mp_steps)]
        )
        self.global_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def forward(self, x, edge_index, edge_attr, x_global, batch):
        """
        Returns:
            structural_emb: [B, hidden_dim]
        """
        x = self.crack_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        g = self.global_encoder(x_global)

        for proc in self.processors:
            x, edge_attr = proc(x, edge_index, edge_attr)

        crack_agg = global_mean_pool(x, batch)  # [B, hidden_dim]
        g = g + self.global_update(torch.cat([g, crack_agg], dim=-1))
        return g


# ============================================================
# Head 2: Temporal (Time series encoder)
# ============================================================

class TemporalHead(nn.Module):
    """Encode time series input (t, T_curve) at each step.

    Supports two backends:
        - 'conv1d': per-step MLP + 1D causal convolution (default)
        - 'lstm':   per-step MLP + single-layer LSTM

    Args:
        time_in_dim:    Input features per time step (default 2: t_norm, T_curve_norm)
        hidden_dim:     Hidden dimension
        temporal_type:  'conv1d' or 'lstm'
        use_conv:       Deprecated, use temporal_type='conv1d' instead
        conv_kernel:    Kernel size for 1D conv
        mlp_layers:     Layers per MLP
    """

    def __init__(self, time_in_dim=4, hidden_dim=64, temporal_type='conv1d',
                 use_conv=True, conv_kernel=5, mlp_layers=2):
        super().__init__()
        if not use_conv and temporal_type == 'conv1d':
            temporal_type = 'mlp_only'
        self.temporal_type = temporal_type
        self.step_mlp = MLP(time_in_dim, hidden_dim, hidden_dim, mlp_layers)

        if temporal_type == 'conv1d':
            self.causal_pad = conv_kernel - 1
            self.conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
        elif temporal_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=1, batch_first=True,
            )
            self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, time_features):
        """
        Args:
            time_features: [B, T, time_in_dim]

        Returns:
            temporal_emb: [B, T, hidden_dim]
        """
        h = self.step_mlp(time_features)  # [B, T, hidden_dim]

        if self.temporal_type == 'conv1d':
            h_t = h.transpose(1, 2)
            h_t = nn.functional.pad(h_t, (self.causal_pad, 0))
            h_conv = self.conv[0](h_t)
            h_conv = self.conv[1](h_conv)
            h_conv = h_conv.transpose(1, 2)
            h_conv = self.conv[2](h_conv)
            h = h + h_conv
        elif self.temporal_type == 'lstm':
            h_lstm, _ = self.lstm(h)
            h = h + self.ln(h_lstm)

        return h


# ============================================================
# CrackFireNet: Full model
# ============================================================

class CrackFireNet(nn.Module):
    """Dual-head model: Structure (GNN) + Temporal (MLP+Conv) -> charring time series.

    Args:
        crack_in_dim:  Crack node features (default 11)
        edge_in_dim:   Edge features (default 5)
        global_in_dim: Global features (default 3)
        time_in_dim:   Time step features (default 2)
        hidden_dim:    Hidden dimension (default 64)
        num_mp_steps:  GNN message passing rounds (default 4)
        mlp_layers:    Layers per MLP block (default 2)
        use_conv:      Use 1D conv in temporal head (default True)
    """

    def __init__(self, crack_in_dim=11, edge_in_dim=5, global_in_dim=3,
                 time_in_dim=4, hidden_dim=64, num_mp_steps=4,
                 mlp_layers=2, use_conv=True, temporal_type='conv1d'):
        super().__init__()

        self.structure_head = StructureHead(
            crack_in_dim, edge_in_dim, global_in_dim,
            hidden_dim, num_mp_steps, mlp_layers,
        )
        self.temporal_head = TemporalHead(
            time_in_dim, hidden_dim, temporal_type=temporal_type,
            use_conv=use_conv, mlp_layers=mlp_layers,
        )

        # Decoder: fused representation -> area ratio prediction
        self.decoder = nn.Sequential(
            MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # area ratio is in [0, 1]
        )

    def forward(self, x, edge_index, edge_attr, x_global, batch,
                time_features):
        """
        Args:
            x:              Crack node features [N_total, crack_in_dim]
            edge_index:     [2, E_total]
            edge_attr:      [E_total, edge_in_dim]
            x_global:       [B, global_in_dim]
            batch:          [N_total] batch assignment
            time_features:  [B, T, time_in_dim]

        Returns:
            area_ratio: [B, T, 1] predicted charring area fraction at each step
        """
        # Handle PyG batching: x_global [B*1, 3] -> [B, 3]
        B = int(batch.max().item()) + 1
        if x_global.size(0) != B:
            x_global = x_global.view(B, -1)

        # Head 1: structural embedding [B, D]
        structural_emb = self.structure_head(
            x, edge_index, edge_attr, x_global, batch
        )

        # Head 2: temporal embedding [B, T, D]
        # PyG DataLoader concatenates time_features along dim 0: [B*T, 2]
        # Need to reshape to [B, T, 2] first
        if time_features.dim() == 2:
            T = time_features.size(0) // B
            time_features = time_features.view(B, T, -1)
        else:
            T = time_features.size(1)

        temporal_emb = self.temporal_head(time_features)

        # Fusion: broadcast structural_emb to T steps, concat with temporal
        structural_expanded = structural_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        fused = torch.cat([structural_expanded, temporal_emb], dim=-1)  # [B, T, 2D]

        # Decode to area ratio
        area_ratio = self.decoder(fused)  # [B, T, 1]

        return area_ratio


# ============================================================
# Utility: generate ISO-834 time features
# ============================================================

def make_time_features(num_steps=60, duration=3600.0, curve_type='iso834',
                       curve_params=None):
    """Generate normalized time features for model input.

    Features per time step (4 dim):
        [0] t_norm:        normalized time t/t_max
        [1] T_norm:        normalized temperature T/1200
        [2] Q_cumul_norm:  cumulative heat exposure (integral of T) / normalization
        [3] dT_dt_norm:    instantaneous heating rate (dT/dt) / normalization

    Args:
        num_steps: number of time steps
        duration: total duration in seconds
        curve_type: fire curve identifier
        curve_params: optional dict of curve-specific parameters

    Returns:
        time_features: [1, num_steps, 4]
    """
    times = torch.linspace(0, duration, num_steps)
    t_norm = times / duration

    T_curve = compute_fire_curve(times, curve_type, curve_params)
    T_norm = T_curve / 1200.0

    dt = duration / max(num_steps - 1, 1)
    Q_cumul = torch.cumsum(T_curve * dt, dim=0)
    Q_cumul_norm = Q_cumul / (1000.0 * duration)

    dT = torch.zeros_like(T_curve)
    dT[1:] = (T_curve[1:] - T_curve[:-1]) / dt
    dT[0] = dT[1]
    dT_dt_norm = dT / 20.0  # ~20 C/s peak for ISO-834

    features = torch.stack([t_norm, T_norm, Q_cumul_norm, dT_dt_norm], dim=-1)
    return features.unsqueeze(0)


def compute_fire_curve(times_s, curve_type='iso834', params=None):
    """Compute temperature [C] at given times [s] for a fire curve.

    Args:
        times_s: tensor of time values in seconds
        curve_type: curve identifier string
        params: optional dict of curve-specific overrides

    Returns:
        temperatures: tensor of temperatures in Celsius
    """
    t_min = times_s / 60.0
    p = params or {}

    if curve_type == 'iso834':
        return 20.0 + 345.0 * torch.log10(8.0 * t_min + 1.0 + 1e-8)

    elif curve_type == 'astm_e119':
        t_hr = times_s / 3600.0
        return 20.0 + 750.0 * (1.0 - torch.exp(-3.79553 * torch.sqrt(t_hr + 1e-8))) \
               + 170.41 * torch.sqrt(t_hr + 1e-8)

    elif curve_type == 'hydrocarbon':
        return 20.0 + 1080.0 * (1.0 - 0.325 * torch.exp(-0.167 * t_min)
                                  - 0.675 * torch.exp(-2.5 * t_min))

    elif curve_type == 'linear':
        k = p.get('rate', 10.0)  # C/min
        return 20.0 + k * t_min

    elif curve_type == 'log_variant':
        a = p.get('a', 345.0)
        b = p.get('b', 8.0)
        return 20.0 + a * torch.log10(b * t_min + 1.0 + 1e-8)

    elif curve_type == 'plateau':
        T_peak = p.get('T_peak', 800.0)
        t_rise = p.get('t_rise_min', 20.0)
        T_iso = 20.0 + 345.0 * torch.log10(8.0 * t_min + 1.0 + 1e-8)
        plateau = torch.full_like(T_iso, T_peak)
        return torch.where(t_min <= t_rise, T_iso, plateau)

    elif curve_type == 'decay':
        t_peak_min = p.get('t_peak_min', 30.0)
        decay_rate = p.get('decay_rate', 0.03)
        T_iso = 20.0 + 345.0 * torch.log10(8.0 * t_min + 1.0 + 1e-8)
        T_at_peak = 20.0 + 345.0 * math.log10(8.0 * t_peak_min + 1.0)
        T_decay = 20.0 + (T_at_peak - 20.0) * torch.exp(-decay_rate * (t_min - t_peak_min))
        return torch.where(t_min <= t_peak_min, T_iso, T_decay)

    elif curve_type == 'perturbed_iso':
        amplitude = p.get('amplitude', 50.0)
        freq = p.get('freq', 0.1)
        T_iso = 20.0 + 345.0 * torch.log10(8.0 * t_min + 1.0 + 1e-8)
        noise = amplitude * torch.sin(2.0 * math.pi * freq * t_min)
        return torch.clamp(T_iso + noise, min=20.0)

    elif curve_type == 'bilinear':
        # Three-phase: linear rise -> constant plateau -> linear decay
        T_peak = p.get('T_peak', 800.0)
        t_rise_min = p.get('t_rise_min', 20.0)
        t_plateau_min = p.get('t_plateau_min', 15.0)
        t_decay_min = p.get('t_decay_min', 25.0)
        T_ambient = p.get('T_ambient', 20.0)
        rate_up = (T_peak - T_ambient) / (t_rise_min + 1e-8)
        t_end_plateau = t_rise_min + t_plateau_min
        T_rise = T_ambient + rate_up * t_min
        T_plat = torch.full_like(t_min, T_peak)
        rate_down = (T_peak - T_ambient) / (t_decay_min + 1e-8)
        T_cool = T_peak - rate_down * (t_min - t_end_plateau)
        T_cool = torch.clamp(T_cool, min=T_ambient)
        return torch.where(t_min <= t_rise_min, T_rise,
               torch.where(t_min <= t_end_plateau, T_plat, T_cool))

    elif curve_type == 'external_fire':
        # EC1 Annex B external fire curve
        return 20.0 + 660.0 * (1.0 - 0.687 * torch.exp(-0.32 * t_min)
                                     - 0.313 * torch.exp(-3.8 * t_min))

    elif curve_type == 'smoldering':
        # Slow smoldering: elevated ambient + very slow linear rise
        T_ambient = p.get('T_ambient', 100.0)
        rate = p.get('rate', 2.0)  # C/min, very slow
        return T_ambient + rate * t_min

    else:
        return 20.0 + 345.0 * torch.log10(8.0 * t_min + 1.0 + 1e-8)
