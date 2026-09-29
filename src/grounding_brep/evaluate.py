import argparse
import json
from pathlib import Path
import torch
import lightning.pytorch as pl
from grounding_brep.runtime import load_config, load_model, dataset
from futurecad.grounding_metrics import GroundingMetrics
from futurecad.grounding_threshold import resolve_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one explicitly selected BRepGround checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-threshold-override",
        action="store_true",
        help="Explicitly permit a calibration experiment using a non-release threshold",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    pl.seed_everything(cfg.test_seed, workers=True)
    model = load_model(cfg, args.checkpoint, args.device).eval()
    declared = getattr(model, "checkpoint_inference_threshold", None)
    threshold = resolve_threshold(
        cfg.model.inference.threshold,
        declared,
        allow_override=args.allow_threshold_override,
    )
    print(
        f"BRepGround inference threshold: {threshold:g}; checkpoint declaration: {declared}",
        flush=True,
    )
    data = dataset(cfg, args.split)
    if not len(data):
        raise ValueError("Evaluation dataset is empty")
    loader = data.get_dataloader(cfg.data.batch_size, False, cfg.data.num_workers)
    metrics = GroundingMetrics(threshold)
    with torch.inference_mode():
        for batch in loader:
            batch["graph"] = batch["graph"].to(args.device)
            probs = torch.sigmoid(model(batch)["pred_scores"].squeeze(-1)).cpu().numpy()
            metrics.update(probs, batch["labels"]["labels"].numpy())
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "samples": len(data),
        "inference_threshold": threshold,
        "checkpoint_inference_threshold": declared,
        "allow_threshold_override": args.allow_threshold_override,
        "metrics": metrics.result(),
        "units": "fractions; multiply by 100 for paper percentages",
        "ranking_ties": "stable descending score order",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
