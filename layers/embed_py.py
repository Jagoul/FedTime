import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = nn.Parameter(torch.randn(1000, d_model))

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding[:x.shape[1], :]
        return self.dropout(x)


class PatchEmbedding_Linear(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout=0.1):
        super(PatchEmbedding_Linear, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        
        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, L, C = x.shape
        
        # Padding
        if L % self.stride != 0:
            length = (L // self.stride + 1) * self.stride
            padding = length - L
            x = F.pad(x, (0, 0, 0, padding), mode='replicate')
        else:
            length = L

        # Patching: [Batch, Input length, Channel] -> [Batch*Channel, Patch number, Patch length]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, patch_num, patch_len]
        x = x.permute(0, 1, 2, 3).contiguous()  # [B, C, patch_num, patch_len]
        x = x.view(-1, x.shape[-2], x.shape[-1])  # [B*C, patch_num, patch_len]
        
        # Input encoding
        x = self.value_embedding(x)  # [B*C, patch_num, d_model]
        
        # Add positional embedding
        x = x + self.position_embedding(x)  # [B*C, patch_num, d_model]
        
        return self.dropout(x)


class PatchEmbedding_Layered(nn.Module):
    """
    Patch embedding with additional layers for complex patterns
    """
    def __init__(self, d_model, patch_len, stride, dropout=0.1, n_layers=2):
        super(PatchEmbedding_Layered, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_layers = n_layers
        
        # Multi-layer patch embedding
        layers = []
        input_dim = patch_len
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, d_model if i == n_layers-1 else d_model//2))
            layers.append(nn.ReLU())
            if i < n_layers - 1:
                layers.append(nn.Dropout(dropout/2))
            input_dim = d_model if i == n_layers-1 else d_model//2
            
        self.value_embedding = nn.Sequential(*layers)
        
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Final dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Patching logic (similar to PatchEmbedding_Linear)
        B, L, C = x.shape
        
        if L % self.stride != 0:
            length = (L // self.stride + 1) * self.stride
            padding = length - L
            x = F.pad(x, (0, 0, 0, padding), mode='replicate')

        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.view(-1, x.shape[-2], x.shape[-1])
        
        # Multi-layer embedding
        x = self.value_embedding(x)
        
        # Add positional embedding
        x = x + self.position_embedding(x)
        
        return self.dropout(x)