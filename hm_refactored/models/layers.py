import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
    
class SideInfoMultiHeadAttention(nn.Module):
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

class VanillaAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        gate_logits = self.score(torch.tanh(self.proj(x)))
        gate = torch.softmax(gate_logits, dim=-1)
        context = torch.sum(gate * x, dim=-1)
        return context, gate

class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced. 
    
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
        
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    
    """
    
    def __init__(self, n_heads, hidden_dim, attribute_hidden_dim, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, fusion_type, max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention"
                "heads (%d)" % (hidden_dim, n_heads)
            )
        # assume hidden_dim = 32
        # assume attribute_hidden_dim = [16, 8, 4]
        self.num_attention_heads = n_heads # 4
        self.attention_head_size = int(hidden_dim / n_heads) # 8
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 32
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_dim] # [4, 2, 1]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size] # [16, 8, 4]
        self.fusion_type = fusion_type
        self.max_len = max_len # 30
        
        self.query = nn.Linear(hidden_dim, self.all_head_size) # (B, N, D) -> (B, N, D)
        self.key = nn.Linear(hidden_dim, self.all_head_size) # (B, N, D) -> (B, N, D)
        self.value = nn.Linear(hidden_dim, self.all_head_size) # (B, N, D) -> (B, N, D)
        
        self.query_p = nn.Linear(hidden_dim, self.all_head_size) # (B, N, D) -> (B, N, D)
        self.key_p = nn.Linear(hidden_dim, self.all_head_size) # (B, N, D) -> (B, N, D)
        
        self.feat_num = feat_num # 3
        
        # ModuleList of nn.Linear which transform dimension of 0th feature: 16 -> 4, 1th feature: 8 -> 2, 2th feature: 4 -> 1
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_dim[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        
        # ModuleList of nn.Linear which transform dimension of 0th feature: 16 -> 4, 1th feature: 8 -> 2, 2th feature: 4 -> 1
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_dim[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)]
        )
        
        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(2 + self.feat_num, 2 + self.feat_num)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    # new_shape을 정의하고 x의 size를 변경
    # last dimension 추가?
    # x의 1, 2번째 dimension을 바꾼 tensor 반환
    
    def transpose_for_scores_attribute(self, x, i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    # i의 역할은?
    # x의 1, 2번째 dimension을 바꾼 tensor 반환
    
    def forward(self, input_tensor, attribute_table, position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))
        
        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))
        
        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        
        attribute_attention_table = []
        
        for i, (attribute_query, attribute_key) in enumerate(
                zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor), i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor), i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
            
        attribute_attention_table = torch.cat(attribute_attention_table, dim=-2)
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size, ))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores = attention_scores.permute(0, 1, 2, 4, 3).contiguous()
            attention_scores, _ = self.fusion_layer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all laeyrs in BertModel forward() function)
        # [batch_size heads seq_len] scores
        # [batch_size 1 1 seq_len]
        
        attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.
    
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                            is the output of the transformer layer.
    
    """
    
    def __init__(self, n_heads, hidden_size, attribute_hidden_size, feat_num, intermediate_size, hidden_dropout_prob,
                 attn_dropout_prob, max_len, fusion_type='sum', layer_norm_eps=1e-12):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size, attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob,
            layer_norm_eps, fusion_type, max_len)
        self.feed_forward = ResidualFF(hidden_size, intermediate_size, hidden_dropout_prob)
        
    def forward(self, hidden_states, attribute_embed, position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attribute_embed, position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output
