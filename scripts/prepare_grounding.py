import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from futurecad.splits import load_splits


def prepare(job):
    import torch
    from dgl.data.utils import save_graphs
    from grounding_brep.preprocessing import build_graph
    from grounding_brep.step_labels import read_step_labels
    from occwl.solid import Solid

    torch.set_num_threads(1)
    root, output, name = job
    sample = read_step_labels(
        root / "steps" / f"{name}.step", root / "labels" / f"{name}.json"
    )
    graph, labels = build_graph(
        Solid(sample["shape"], allow_compound=True), sample["labels"]
    )
    for fields in (graph.ndata, graph.edata):
        if any(not torch.isfinite(fields[key]).all().item() for key in fields):
            raise ValueError(f"Non-finite graph features for {name}")
    save_graphs(str(output / "graph" / f"{name}.bin"), [graph])
    (output / "labels" / f"{name}.json").write_text(
        json.dumps(labels), encoding="utf-8"
    )
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Build the BRepGround model cache from STEP and primitive labels."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/grounding"))
    parser.add_argument("--output", type=Path, default=Path("datasets/grounding_cache"))
    parser.add_argument(
        "--splits", type=Path, default=Path("data/splits/grounding.json")
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.dataset.resolve() == args.output.resolve():
        parser.error("--output must differ from --dataset")
    if args.workers < 1:
        parser.error("--workers must be positive")
    splits = load_splits(args.splits)
    names = (
        [name for group in splits.values() for name in group]
        if args.split == "all"
        else splits[args.split]
    )
    if args.limit:
        names = names[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in ["graph", "labels"]:
        (args.output / kind).mkdir(exist_ok=True)
        if any(
            (
                args.output / kind / f"{name}.{'bin' if kind == 'graph' else 'json'}"
            ).exists()
            for name in names
        ):
            parser.error(
                "Output already contains selected states; choose a new output directory"
            )
    jobs = [(args.dataset, args.output, name) for name in names]
    if args.workers == 1:
        for i, _ in enumerate(map(prepare, jobs), 1):
            print(f"{i}/{len(names)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, _ in enumerate(pool.map(prepare, jobs), 1):
                if i % 100 == 0 or i == len(names):
                    print(f"{i}/{len(names)}", flush=True)


if __name__ == "__main__":
    main()
