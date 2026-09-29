import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InGraphTokenContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_proto=0.3):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        self.lambda_proto = lambda_proto

    def forward(self, uv_states, text_states, gt_labels, text_token_mask):
        B, L_uv, d = uv_states.shape

        uv_valid = gt_labels != -1
        uv_pos = gt_labels == 1
        sample_has_pos = uv_pos.any(dim=1)

        if not sample_has_pos.any():
            return uv_states.sum() * 0.0

        uv = F.normalize(uv_states, dim=-1)
        txt = F.normalize(text_states, dim=-1)

        logit_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()

        sim = torch.einsum("btd,bud->btu", txt, uv) * logit_scale

        mask_val = torch.finfo(sim.dtype).min
        sim = sim.masked_fill(text_token_mask[:, :, None], mask_val)

        attn = F.softmax(sim.detach(), dim=1)
        scores_uv = (attn * sim).sum(dim=1)

        scores_uv_masked = scores_uv.masked_fill(~uv_valid, mask_val)

        pos_scores = scores_uv_masked.masked_fill(~uv_pos, mask_val)
        log_num = torch.logsumexp(pos_scores, dim=1)

        log_den = torch.logsumexp(scores_uv_masked, dim=1)

        loss_per_sample = -(log_num - log_den)
        loss_ground = loss_per_sample[sample_has_pos].mean()

        txt_mask = ~text_token_mask
        txt_mask_sum = txt_mask.sum(dim=1, keepdim=True).clamp(min=1)
        text_global = (text_states * txt_mask[..., None]).sum(dim=1) / txt_mask_sum
        text_global = F.normalize(text_global, dim=-1)

        uv_pos_expanded = uv_pos[..., None].float()
        pos_sum = (uv_states * uv_pos_expanded).sum(dim=1)
        pos_count = uv_pos.sum(dim=1, keepdim=True).clamp(min=1)
        geom_proto = pos_sum / pos_count
        geom_proto = F.normalize(geom_proto, dim=-1)

        sim_proto = logit_scale * (text_global * geom_proto).sum(dim=1)

        loss_proto_per_sample = -F.logsigmoid(sim_proto)
        loss_proto = loss_proto_per_sample[sample_has_pos].mean()

        loss = loss_ground + self.lambda_proto * loss_proto

        return loss
