import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from grounding_brep.models.losses.contrastive import InGraphTokenContrastiveLoss
from grounding_brep.models.losses.weighted_bce import ReferenceLoss
from grounding_brep.models.uv_net.encoder import UVNetModel
from grounding_brep.models.text import BERTWordEncoder
from grounding_brep.models.fusion import FusionModule
from grounding_brep.models.adapters.geometry import UVAdaptiveLayer
from grounding_brep.models.adapters.feature import AdaptiveLayer


class GroundingBRep(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.uv_encoder = UVNetModel(
            crv_in_channels=cfg.model.uv.crv_in_channels,
            surf_in_channels=cfg.model.uv.surf_in_channels,
            crv_emb_dim=cfg.model.uv.crv_emb_dim,
            srf_emb_dim=cfg.model.uv.srf_emb_dim,
            graph_emb_dim=cfg.model.uv.graph_emb_dim,
        )

        self.uv_adaptive = UVAdaptiveLayer(
            d_model=cfg.model.uv.graph_emb_dim,
            h=cfg.model.uv.heads,
            out_dim=cfg.model.uv.out_dim,
            dropout=cfg.model.uv.dropout,
            ffn_dim=cfg.model.uv.ffn_dim,
        )

        self.text_encoder = BERTWordEncoder(
            bert_model_path=cfg.model.text.bert_path,
            h=cfg.model.text.heads,
            initialize_pretrained=cfg.model.text.get("initialize_pretrained", True),
        )
        for p in self.text_encoder.bert.parameters():
            p.requires_grad = False

        self.text_adaptive = AdaptiveLayer(
            d_model=cfg.model.text.d_model,
            h=cfg.model.text.heads,
            out_dim=cfg.model.text.out_dim,
            dropout=cfg.model.text.dropout,
            ffn_dim=cfg.model.text.ffn_dim,
        )

        self.fusion = FusionModule(
            d_model=cfg.model.fusion.d_model,
            num_heads=cfg.model.fusion.heads,
            depth=cfg.model.fusion.depth,
            head_mode=cfg.model.fusion.match_mode,
            dropout=cfg.model.fusion.dropout,
            ffn_dim=cfg.model.fusion.ffn_dim,
        )

        if cfg.model.loss.use_contrastive:
            print("Using In-Graph Token Contrastive Loss")
            self.contrastive_loss = InGraphTokenContrastiveLoss(
                temperature=cfg.model.loss.contrastive_temperature,
            )
        else:
            print("Not using Contrastive Loss")
            self.contrastive_loss = None
        self.reference_loss = ReferenceLoss()

    def forward(self, batch):
        output_dict = {}
        node_embs, edge_embs = self.uv_encoder(batch["graph"])

        edge_embs = 0.5 * (edge_embs[0::2] + edge_embs[1::2])
        edge_counts_directed = batch["graph"].batch_num_edges().tolist()
        edge_counts = [c // 2 for c in edge_counts_directed]
        node_counts = batch["graph"].batch_num_nodes().tolist()
        uv_out = self.uv_adaptive(node_embs, edge_embs, node_counts, edge_counts)

        uv_states = uv_out["hidden_states"]
        uv_mask = uv_out["attention_mask"]

        bert_out = self.text_encoder({"text": batch["query"]})
        text_adap_out = self.text_adaptive(bert_out, {})
        text_states = text_adap_out["hidden_states"]
        text_mask = text_adap_out["attention_mask"]

        diag = torch.diagonal(text_mask, dim1=-2, dim2=-1)
        text_token_mask = diag.any(dim=1)
        output_dict["text_token_mask"] = text_token_mask

        pred_scores = self.fusion(
            uv_states=uv_states,
            uv_mask=uv_mask,
            text_states=text_states,
            text_mask=text_mask,
        )

        output_dict["pred_scores"] = pred_scores
        output_dict["uv_states"] = uv_states
        output_dict["text_states"] = text_states

        return output_dict

    def _compute_loss(self, batch, output_dict):
        losses = {}
        gt_labels = batch["labels"]["labels"]
        if self.contrastive_loss is not None:
            losses["contrastive"] = self.contrastive_loss(
                uv_states=output_dict["uv_states"],
                text_states=output_dict["text_states"],
                gt_labels=gt_labels,
                text_token_mask=output_dict["text_token_mask"],
            )

        losses["reference"] = self.reference_loss(
            pred_scores=output_dict["pred_scores"],
            gt_labels=gt_labels,
        )

        losses["total"] = sum(losses.values())
        return losses

    def training_step(self, batch, idx):
        output_dict = self(batch)
        loss_dict = self._compute_loss(batch, output_dict)
        bs = len(batch["query"])
        self.log_dict(
            {f"train_loss/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=False,
            batch_size=bs,
            sync_dist=False,
            rank_zero_only=True,
        )

        return loss_dict["total"]

    def validation_step(self, batch, idx):
        output_dict = self(batch)
        loss_dict = self._compute_loss(batch, output_dict)
        bs = len(batch["query"])

        self.log_dict(
            {f"val_loss/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            batch_size=bs,
            sync_dist=True,
        )

        pred = self._predict_labels(output_dict)
        gt = batch["labels"]["labels"]
        pred = pred.view(-1)
        gt = gt.view(-1)
        assert pred.shape == gt.shape
        mask = gt != -1
        threshold = self.cfg.model.inference.threshold
        pred = pred[mask] >= threshold
        gt = gt[mask] == 1

        tp = ((pred == 1) & (gt == 1)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()
        tn = ((pred == 0) & (gt == 0)).sum()

        self.log("val_tp", tp.float(), sync_dist=True, reduce_fx="sum")
        self.log("val_fp", fp.float(), sync_dist=True, reduce_fx="sum")
        self.log("val_fn", fn.float(), sync_dist=True, reduce_fx="sum")
        self.log("val_tn", tn.float(), sync_dist=True, reduce_fx="sum")

        logits = output_dict["pred_scores"].squeeze(-1)
        probs = torch.sigmoid(logits)
        labels = batch["labels"]["labels"]

        k_list = [1, 3, 5, 10]

        hit_counts = {k: 0.0 for k in k_list}
        recall_sums = {k: 0.0 for k in k_list}
        valid_samples = 0

        for i in range(bs):
            sample_probs = probs[i]
            sample_labels = labels[i]

            valid_mask = sample_labels != -1
            if valid_mask.sum() == 0:
                continue

            sample_probs = sample_probs[valid_mask]
            sample_labels = sample_labels[valid_mask]

            gt_positive_indices = (sample_labels == 1).nonzero(as_tuple=True)[0]
            num_gt = len(gt_positive_indices)

            if num_gt == 0:
                continue

            valid_samples += 1
            num_valid = len(sample_probs)

            sorted_indices = torch.argsort(sample_probs, descending=True)

            for k in k_list:
                top_k = min(k, num_valid)
                top_k_indices = sorted_indices[:top_k]
                top_k_set = set(top_k_indices.cpu().tolist())
                gt_set = set(gt_positive_indices.cpu().tolist())

                hit = len(top_k_set & gt_set) > 0
                hit_counts[k] += float(hit)

                recall = len(top_k_set & gt_set) / num_gt
                recall_sums[k] += recall

        for k in k_list:
            self.log(
                f"val_hit@{k}_sum",
                float(hit_counts[k]),
                sync_dist=True,
                reduce_fx="sum",
            )
            self.log(
                f"val_recall@{k}_sum",
                float(recall_sums[k]),
                sync_dist=True,
                reduce_fx="sum",
            )

        self.log(
            "val_valid_samples", float(valid_samples), sync_dist=True, reduce_fx="sum"
        )

    def on_validation_epoch_end(self):
        tp = self.trainer.callback_metrics["val_tp"]
        fp = self.trainer.callback_metrics["val_fp"]
        fn = self.trainer.callback_metrics["val_fn"]
        tn = self.trainer.callback_metrics["val_tn"]

        eps = 1e-6
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        self.log_dict(
            {
                "val/accuracy": acc,
                "val/precision": prec,
                "val/recall": rec,
                "val/f1": f1,
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        valid_samples = self.trainer.callback_metrics["val_valid_samples"]
        k_list = [1, 3, 5, 10]
        for k in k_list:
            hit_sum = self.trainer.callback_metrics[f"val_hit@{k}_sum"]
            recall_sum = self.trainer.callback_metrics[f"val_recall@{k}_sum"]
            hit_at_k = hit_sum / valid_samples * 100
            recall_at_k = recall_sum / valid_samples * 100
            self.log(
                f"val_hit@{k}", hit_at_k, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.log(
                f"val_recall@{k}",
                recall_at_k,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

    def _predict_labels(self, output_dict):
        logits = output_dict["pred_scores"].squeeze(-1)
        probs = torch.sigmoid(logits)
        threshold = self.cfg.model.inference.threshold
        return (probs >= threshold).float()

    def configure_optimizers(self):
        lr = self.cfg.optim.lr
        wd = self.cfg.optim.weight_decay

        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=lr * 0.05,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def setup(self, stage=None):
        if stage == "fit" and self.trainer.is_global_zero:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

            def to_human(numel):
                if numel >= 1_000_000_000:
                    return f"{numel / 1e9:.2f}B"
                elif numel >= 1_000_000:
                    return f"{numel / 1e6:.2f}M"
                else:
                    return str(numel)

            print("\n================= GroundingBRep Model Summary =================")
            print(
                f"Total parameters:      {total_params:,}  ({to_human(total_params)})"
            )
            print(
                f"Trainable parameters:  {trainable_params:,}  ({to_human(trainable_params)})"
            )
            print("---------------------------------------------------------------")
            print("Trainable Submodules:")
            for name, module in self.named_children():
                has_trainable = any(p.requires_grad for p in module.parameters())
                print(f"  {name:<20} trainable={has_trainable}")
            print("===============================================================\n")

            print(
                "\n================= Trainable Parameter Breakdown (M / B units) ================="
            )
            for name, module in self.named_children():
                module_total = sum(p.numel() for p in module.parameters())
                module_trainable = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )

                print(f"[{name}]")
                print(
                    f"  total params:     {module_total:,}  ({to_human(module_total)})"
                )
                print(
                    f"  trainable params: {module_trainable:,}  ({to_human(module_trainable)})"
                )
            print(
                "================================================================================\n"
            )
