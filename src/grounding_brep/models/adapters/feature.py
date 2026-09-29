import torch.nn as nn

from grounding_brep.models.common.attention import MultiHeadAttention
from grounding_brep.models.common.ffn import FeedForward


class AdaptiveLayer(nn.Module):
    def __init__(
        self,
        d_model,
        h,
        out_dim,
        dropout=0.1,
        ffn_dim=2048,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, dim_ff=ffn_dim, dropout=dropout)

        self.has_downsampler = d_model != out_dim
        if self.has_downsampler:
            self.downsampler = nn.Linear(d_model, out_dim)

    def forward(self, data_dict, output_dict):
        hidden_states = data_dict["hidden_states"]
        attention_mask = data_dict.get("attention_mask", None)

        residual = self.norm1(hidden_states)

        attn_output = self.self_attn(
            residual,
            residual,
            residual,
            attention_mask=attention_mask,
            attention_weights=None,
            way="add",
        )

        hidden_states = hidden_states + self.dropout1(attn_output)

        residual = self.norm2(hidden_states)
        hidden_states = hidden_states + self.dropout2(self.ffn(residual))

        if self.has_downsampler:
            hidden_states = self.downsampler(hidden_states)

        output_dict["hidden_states"] = hidden_states
        output_dict["attention_mask"] = attention_mask

        return output_dict
