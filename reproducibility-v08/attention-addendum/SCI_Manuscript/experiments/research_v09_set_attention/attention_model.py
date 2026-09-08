"""Fixed interaction-capable set control, proposed after reviewing v08 results.

Four shared, position-free self-attention blocks replace independent-node-only
processing. No raw edge input is used; all observations are the same node,
global and prescribed-time tensors supplied to the original set controls.
"""
from pathlib import Path
import sys
import math
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import Surrogate, mean_pool

MODEL_NAME = 'set_attention'
MODEL_VERSION = 'set-attention-h64-heads4-blocks4-ff104-v1'
ARCHITECTURE = {'hidden_dim': 64, 'heads': 4, 'attention_blocks': 4,
                'feedforward_dim': 104, 'dropout': 0, 'position_encoding': False,
                'activation': 'ReLU', 'normalization': 'pre-LayerNorm',
                'pooling': 'arithmetic mean of valid nodes', 'self_attention_diagonal': True,
                'raw_edge_attributes_used': False, 'allowed_crack_counts': [1, 15]}


class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attention = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.output = nn.Linear(64, 64)
        self.norm_feedforward = nn.LayerNorm(64)
        self.feedforward = nn.Sequential(nn.Linear(64, 104), nn.ReLU(), nn.Linear(104, 64))

    def forward(self, h, valid):
        b, n, _ = h.shape
        qkv = self.qkv(self.norm_attention(h)).reshape(b, n, 3, 4, 16)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(16.)
        scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
        message = torch.matmul(scores.softmax(dim=-1), v).transpose(1, 2).reshape(b, n, 64)
        h = h + self.output(message)
        h = h + self.feedforward(self.norm_feedforward(h))
        return h.masked_fill(~valid[:, :, None], 0.)


class InteractionSetStructure(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.crack_encoder = original.crack_encoder
        self.global_encoder = original.global_encoder
        self.global_update = original.global_update
        self.blocks = nn.ModuleList([SelfAttentionBlock() for _ in range(4)])

    def forward(self, x, edge_index, edge_attr, x_global, batch):
        if x.ndim != 2 or x.shape[1] != 11 or x_global.ndim != 2 or x_global.shape[1] != 3:
            raise ValueError('Require the frozen 11-node/3-global feature interface')
        b = x_global.shape[0]
        if b == 0 or batch.ndim != 1 or len(batch) != len(x):
            raise ValueError('Require a nonempty batch with one graph identity per node')
        if bool(((batch < 0) | (batch >= b)).any()):
            raise ValueError('Node graph identity outside the supplied batch')
        counts = torch.bincount(batch, minlength=b)
        if bool(((counts < 1) | (counts > 15)).any()):
            raise ValueError('Scientific scope is one to fifteen cracks; empty/out-of-scope queries are rejected')
        encoded = self.crack_encoder(x)
        dense = encoded.new_zeros((b, 15, 64))
        for g in range(b):
            nodes = encoded[batch == g]
            dense[g, :len(nodes)] = nodes
        valid = torch.arange(15, device=x.device)[None, :] < counts[:, None]
        for block in self.blocks:
            dense = block(dense, valid)
        pooled = dense.sum(dim=1) / counts[:, None]
        global_state = self.global_encoder(x_global)
        return global_state + self.global_update(torch.cat([global_state, pooled], -1))


class SetAttentionSurrogate(Surrogate):
    def __init__(self):
        # The shared branches and original small-node map keep their exact code.
        # As with the matched-set constructor, added structural parameters are
        # initialized after the ordinary set's temporal/decoder modules.
        super().__init__(mode='deepsets')
        self.structure_head = InteractionSetStructure(self.structure_head)
        self.mode = MODEL_NAME

    def forward(self, x, edge_index, edge_attr, x_global, batch, time_features):
        b = x_global.shape[0]
        if time_features.shape[-1] != 4 or time_features.numel() != b * 61 * 4:
            raise ValueError('Require the frozen 61-time by 4 prescribed-exposure input')
        return super().forward(x, edge_index, edge_attr, x_global, batch, time_features)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
