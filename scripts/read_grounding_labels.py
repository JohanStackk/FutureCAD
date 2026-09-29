import argparse
import json
from pathlib import Path

from grounding_brep.step_labels import read_step_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/grounding"))
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    if not args.id or "/" in args.id or "\\" in args.id or args.id in {".", ".."}:
        parser.error("--id must be a state ID, not a path")
    sample = read_step_labels(
        args.dataset / "steps" / f"{args.id}.step",
        args.dataset / "labels" / f"{args.id}.json",
    )
    labels = sample["labels"]
    print(json.dumps({
        "id": args.id,
        "query": labels["query"],
        "selected_face_indices": [i for i, x in enumerate(labels["face_labels"]) if x],
        "selected_edge_indices": [i for i, x in enumerate(labels["edge_labels"]) if x],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
