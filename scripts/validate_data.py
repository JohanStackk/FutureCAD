import argparse
import json
from pathlib import Path

from futurecad.code_metrics import GROUPS
from futurecad.splits import load_splits


def validate(root):
    futurecad = load_splits(root / "splits/futurecad.json")
    grounding = load_splits(root / "splits/grounding.json")
    all_ids = set().union(*map(set, futurecad.values()))
    advanced = set(json.loads((root / "splits/advanced_ids.json").read_text()))
    counts = json.loads(
        (root / "evaluation/futurecad_operation_counts.json").read_text()
    )
    evaluation_ids = set(futurecad["val"]) | set(futurecad["test"])
    if evaluation_ids != counts.keys():
        raise ValueError(
            "Operation-count IDs must exactly match FutureCAD val/test IDs"
        )
    if advanced - all_ids:
        raise ValueError("Advanced IDs are outside the FutureCAD split")
    required = set().union(*map(set, GROUPS.values()))
    for data_id, operations in counts.items():
        for key in required:
            value = operations.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid operation count: FutureCAD/{data_id}/{key}")
    parents = {split: {name[:8] for name in ids} for split, ids in grounding.items()}
    return {
        "duplicate_and_exact_id_overlap": "none",
        "futurecad_counts": {k: len(v) for k, v in futurecad.items()},
        "grounding_state_counts": {k: len(v) for k, v in grounding.items()},
        "futurecad_advanced": {k: len(set(v) & advanced) for k, v in futurecad.items()},
        "futurecad_evaluation_count_coverage": len(evaluation_ids),
        "grounding_parent_id_overlap": {
            f"{a}/{b}": len(parents[a] & parents[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "note": "Original parent-model overlap is retained; these are not family-disjoint splits.",
    }


def main():
    parser = argparse.ArgumentParser(description='Validate dataset splits and operation counts.')
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(validate(args.data_dir), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
