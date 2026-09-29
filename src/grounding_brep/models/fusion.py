import torch.nn as nn
from grounding_brep.models.common.attention import MultiHeadAttention
from grounding_brep.models.common.ffn import FeedForward


class MatchHead(nn.Module):
    def __init__(self, d_model, mode="linear"):
        super().__init__()
        if mode == "linear":
            self.use_conv = False
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model), nn.PReLU(), nn.Linear(d_model, 1)
            )
        elif mode == "conv":
            self.use_conv = True
            self.mlp = nn.Sequential(
                nn.Conv1d(d_model, d_model, 1),
                nn.BatchNorm1d(d_model),
                nn.PReLU(),
                nn.Conv1d(d_model, d_model, 1),
                nn.BatchNorm1d(d_model),
                nn.PReLU(),
                nn.Conv1d(d_model, 1, 1),
            )
        else:
            raise NotImplementedError(mode)

    def forward(self, x):
        if not self.use_conv:
            return self.mlp(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.mlp(x)
            return x.permute(0, 2, 1)


class FusionModule(nn.Module):
    def __init__(
        self, d_model, num_heads, depth, head_mode="linear", dropout=0.1, ffn_dim=2048
    ):
        super().__init__()

        self.depth = depth
        self.self_attn = nn.ModuleList(
            [
                MultiHeadAttention(d_model=d_model, h=num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.cross_attn = nn.ModuleList(
            [
                MultiHeadAttention(d_model=d_model, h=num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.ffn = nn.ModuleList(
            [
                FeedForward(d_model, dim_ff=ffn_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.norm_self = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.norm_cross = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])

        self.match = MatchHead(d_model, mode=head_mode)

    def forward(self, uv_states, uv_mask, text_states, text_mask):
        hidden = uv_states

        for i in range(self.depth):
            x = self.norm_self[i](hidden)

            sa_out = self.self_attn[i](
                x, x, x, attention_mask=uv_mask, attention_weights=None, way="add"
            )
            hidden = hidden + sa_out

            x = self.norm_cross[i](hidden)

            text_kv_mask = text_mask[:, :, 0:1, :]
            cross_mask = text_kv_mask.expand(-1, -1, uv_states.size(1), -1)

            ca_out = self.cross_attn[i](
                x,
                text_states,
                text_states,
                attention_mask=cross_mask,
                attention_weights=None,
                way="add",
            )
            hidden = hidden + ca_out

            x = self.norm_ffn[i](hidden)
            hidden = hidden + self.ffn[i](x)

        pred_scores = self.match(hidden)

        return pred_scores
