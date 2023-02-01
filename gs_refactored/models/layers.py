import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class EncoderLayer(nn.Module):
    """
    Ref : https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """
    def __init__(self, hidden_dim, n_heads=8, drop_out=0.0):
        super().__init__()
        hidden_dims = hidden_dim * 2

        self.multi_head_attn = MultiheadAttention(hidden_dim, n_heads, drop_out)
        self.feed_forward = ResidualFF(hidden_dim, hidden_dims, drop_out)

        self.scores = None

    def forward(self, x, mask=None, predefined_attn=None):
        out, scores = self.multi_head_attn(x, mask, predefined_attn)
        intermediate = out
        out = self.feed_forward(out)

        self.scores = scores

        return out

class ScaledDotProductAttention(nn.Module):
    """
    Ref : https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """
    def __init__(self, dk, attn_dropout=0.1):
        super().__init__()
        self.dk = dk**0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, predefined_attn=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.dk

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        if predefined_attn != None:
            num_heads = attn.shape[1]
            attn = torch.cat([attn[:,:(num_heads // 2),:,:], predefined_attn[:,(num_heads//2):,:,:]], dim=1)
            
        out = torch.matmul(attn, v)

        return out, attn


class MultiheadAttention(nn.Module):
    """
    Ref : https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """
    def __init__(self, hidden_dim, n_heads=8, drop_out=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dk = hidden_dim // n_heads

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn = ScaledDotProductAttention(dk=hidden_dim // n_heads)

        self.dropout = nn.Dropout(drop_out)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None, predefined_attn=None):
        batch, seq_len, _ = x.size()
        residual = x
        x = self.ln(x)

        q, k, v = self.q(x), self.k(x), self.v(x)

        q_out = q.view(batch, seq_len, self.n_heads, self.dk).transpose(1, 2)
        k_out = k.view(batch, seq_len, self.n_heads, self.dk).transpose(1, 2)
        v_out = v.view(batch, seq_len, self.n_heads, self.dk).transpose(1, 2)

        out, scores = self.attn(q_out, k_out, v_out, mask=mask, predefined_attn=predefined_attn)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.out(out)
        out = self.dropout(out)

        out += residual
        # out = self.ln(out)

        return out, scores

class ResidualFF(nn.Module):
    """
    Ref : https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """
    def __init__(self, in_dims, hid_dims, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, in_dims)

        self.ln = nn.LayerNorm(in_dims, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.ln(x)

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)

        out += residual
        # out = self.ln(out)

        return out

class PositionalEncoding(nn.Module):
    """
    Ref : https://github.com/cpm0722/transformer_pytorch/blob/main/models/embedding/positional_encoding.py
    """
    def __init__(self, hidden_dim, max_len=512, device=torch.device("cpu")):
        super().__init__()
        encoding = self._get_sinusoidal_embedding(hidden_dim, max_len)

        self.encoding = encoding.unsqueeze(0).to(device)
        self.encoding.requires_grad = False

    def _get_sinusoidal_embedding(self, hidden_dim, max_len):
        encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        sinusoidal = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        encoding[:, 0::2] = torch.sin(position * sinusoidal)
        encoding[:, 1::2] = torch.cos(position * sinusoidal)

        return encoding

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out

class DeepFC(nn.Module):
    def __init__(self, in_dims, hid_dims):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, in_dims)
        self.fc3 = nn.Linear(hid_dims, in_dims)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out