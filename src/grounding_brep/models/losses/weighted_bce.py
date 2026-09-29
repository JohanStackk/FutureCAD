import torch
import torch.nn as nn


class ReferenceLoss(nn.Module):
    def __init__(self, pos_weight=20.0):
        super().__init__()
        self.pos_weight = pos_weight
        print(f"ReferenceLoss initialized with pos_weight={self.pos_weight}")

    def forward(self, pred_scores, gt_labels):
        B, L, _ = pred_scores.shape

        valid_mask = gt_labels != -1
        if not valid_mask.any():
            return pred_scores.sum() * 0.0

        target = torch.clamp(gt_labels, min=0).float()
        pred_scores = pred_scores.squeeze(-1)

        per_token_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_scores,
            target,
            reduction="none",
            pos_weight=torch.tensor([self.pos_weight]).to(pred_scores.device),
        )

        masked_loss = per_token_loss * valid_mask.float()
        loss = masked_loss.sum() / valid_mask.sum()

        return loss
