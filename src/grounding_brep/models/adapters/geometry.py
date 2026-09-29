import torch
from torch import nn

from grounding_brep.models.adapters.feature import AdaptiveLayer


class UVAdaptiveLayer(nn.Module):
    def __init__(self, d_model, h, out_dim, dropout=0.1, ffn_dim=2048):
        super().__init__()

        self.segment_embed = nn.Embedding(2, d_model)

        self.adaptive_layer = AdaptiveLayer(
            d_model=d_model, h=h, out_dim=out_dim, dropout=dropout, ffn_dim=ffn_dim
        )

        self.d_model = d_model
        self.out_dim = out_dim

    def forward(self, node_embs, edge_embs, node_counts, edge_counts):
        device = next(self.parameters()).device

        B = len(node_counts)

        node_split = torch.split(node_embs, node_counts, dim=0)
        edge_split = torch.split(edge_embs, edge_counts, dim=0)

        seqs = []
        lengths = []
        for n_emb, e_emb in zip(node_split, edge_split):
            seq = torch.cat([n_emb, e_emb], dim=0)

            seg_ids = torch.cat(
                [
                    torch.zeros(n_emb.size(0), dtype=torch.long, device=device),
                    torch.ones(e_emb.size(0), dtype=torch.long, device=device),
                ]
            )

            seq = seq + self.segment_embed(seg_ids)

            seqs.append(seq)
            lengths.append(seq.size(0))

        L_max = max(lengths)
        hidden_states = torch.zeros((B, L_max, self.d_model), device=device)
        attention_mask = torch.ones(
            (B, 1, L_max, L_max), dtype=torch.bool, device=device
        )

        for i, seq in enumerate(seqs):
            L = seq.size(0)
            hidden_states[i, :L] = seq
            attention_mask[i, :, :L, :L] = False
            attention_mask[i, :, L:, L:] = False

        data_dict = {"hidden_states": hidden_states, "attention_mask": attention_mask}
        output_dict = {}

        output_dict = self.adaptive_layer(data_dict, output_dict)

        output_dict["attention_mask"] = attention_mask
        output_dict["lengths"] = lengths

        return output_dict
