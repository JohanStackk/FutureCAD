from torch import nn
import torch


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.zeros_(self.fc_q.bias)
        nn.init.zeros_(self.fc_k.bias)
        nn.init.zeros_(self.fc_v.bias)
        nn.init.zeros_(self.fc_o.bias)

    def forward(
        self,
        queries,
        keys,
        values,
        attention_mask=None,
        attention_weights=None,
        way="add",
    ):

        B, Nq = queries.shape[:2]
        Nk = keys.shape[1]

        q = self.fc_q(queries).view(B, Nq, self.h, self.d_k).transpose(1, 2)
        k = self.fc_k(keys).view(B, Nk, self.h, self.d_k).transpose(1, 2)
        v = self.fc_v(values).view(B, Nk, self.h, self.d_v).transpose(1, 2)

        att_bias = None
        if attention_weights is not None:
            if way == "mul":
                att_bias = torch.log(attention_weights.clamp(min=1e-6))
            elif way == "add":
                att_bias = attention_weights
            else:
                raise NotImplementedError(way)

        if attention_mask is not None:
            mask_bias = torch.zeros_like(attention_mask, dtype=q.dtype)
            mask_bias = mask_bias.masked_fill(attention_mask, float("-inf"))
            att_bias = mask_bias if att_bias is None else (att_bias + mask_bias)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=att_bias, dropout_p=0.0, is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, Nq, self.h * self.d_v)

        return self.fc_o(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"
        self.d_k = d_model // h
        self.d_v = self.d_k

        self.attention = ScaledDotProductAttention(
            d_model=d_model, d_k=self.d_k, d_v=self.d_v, h=h
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries,
        keys,
        values,
        attention_mask=None,
        attention_weights=None,
        way="add",
    ):
        out = self.attention(
            queries, keys, values, attention_mask, attention_weights, way
        )
        out = self.dropout(out)
        return self.layer_norm(queries + out)
