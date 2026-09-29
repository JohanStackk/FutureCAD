import torch
from torch import nn
import grounding_brep.models.uv_net.encoders as UVEncoders


class UVNetModel(nn.Module):
    def __init__(
        self,
        crv_in_channels=6,
        surf_in_channels=7,
        crv_emb_dim=128,
        srf_emb_dim=128,
        graph_emb_dim=256,
    ):
        super().__init__()
        assert crv_emb_dim == srf_emb_dim
        self.crv_emb_dim = crv_emb_dim
        self.srf_emb_dim = srf_emb_dim
        self.graph_emb_dim = graph_emb_dim
        diff = graph_emb_dim - srf_emb_dim
        self.curv_encoder = UVEncoders.UVNetCurveEncoder(
            in_channels=crv_in_channels, output_dims=crv_emb_dim
        )
        self.surf_encoder = UVEncoders.UVNetSurfaceEncoder(
            in_channels=surf_in_channels, output_dims=srf_emb_dim
        )
        face_attr_dim = diff // 4

        self.face_area_fc = nn.Linear(1, face_attr_dim)
        self.face_type_fc = nn.Linear(11, face_attr_dim)
        self.face_centroid_fc = nn.Linear(3, face_attr_dim)
        self.face_rational_fc = nn.Linear(1, face_attr_dim)

        self.edge_type_fc = nn.Linear(9, diff // 2)
        self.edge_len_fc = nn.Linear(1, diff // 4)
        self.edge_conv_fc = nn.Linear(3, diff // 4)

        self.graph_encoder = UVEncoders.UVNetGraphEncoder(
            input_dim=graph_emb_dim,
            input_edge_dim=graph_emb_dim,
            hidden_dim=graph_emb_dim,
        )

    def forward(self, batched_graph):
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]

        input_crv_feat = input_crv_feat.permute(0, 2, 1)
        input_srf_feat = input_srf_feat.permute(0, 3, 1, 2)

        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)

        face_area = self.face_area_fc(batched_graph.ndata["area"])
        face_type = self.face_type_fc(batched_graph.ndata["ftype"])
        face_centroid = self.face_centroid_fc(batched_graph.ndata["centroid"])
        face_rational = self.face_rational_fc(batched_graph.ndata["rational"])

        face_attr_emb = torch.cat(
            [face_area, face_type, face_centroid, face_rational], dim=-1
        )

        edge_type = self.edge_type_fc(batched_graph.edata["etype"])
        edge_len = self.edge_len_fc(batched_graph.edata["length"])
        edge_conv = self.edge_conv_fc(batched_graph.edata["convexity"])

        edge_attr_emb = torch.cat([edge_type, edge_len, edge_conv], dim=-1)

        face_emb = torch.cat([hidden_srf_feat, face_attr_emb], dim=-1)
        edge_emb = torch.cat([hidden_crv_feat, edge_attr_emb], dim=-1)

        node_embs, edge_embs = self.graph_encoder(batched_graph, face_emb, edge_emb)
        return node_embs, edge_embs
