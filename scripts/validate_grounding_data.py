import argparse
import json
from pathlib import Path


def load_split(path):
    splits = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits")
    seen = set()
    for key, ids in splits.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Empty or invalid split: {key}")
        for name in ids:
            if (not isinstance(name, str) or not name or "/" in name
                    or "\\" in name or name in {".", ".."} or name in seen):
                raise ValueError(f"Invalid or duplicate state ID: {name!r}")
            seen.add(name)
    return splits


def validate_state(root, name):
    import torch
    from dgl.data.utils import load_graphs

    root = Path(root)
    data = json.loads((root / "labels" / f"{name}.json").read_text(encoding="utf-8"))
    queries, labels = data["query"], data["labels"]
    if not isinstance(queries, list) or not queries or any(
        not isinstance(q, str) or not q.strip() for q in queries
    ):
        raise ValueError(f"Invalid queries: {name}")
    if not isinstance(labels, list) or not labels or any(x not in (0, 1) for x in labels):
        raise ValueError(f"Invalid binary labels: {name}")
    graphs, _ = load_graphs(str(root / "graph" / f"{name}.bin"))
    if len(graphs) != 1:
        raise ValueError(f"Expected one graph: {name}")
    graph = graphs[0]
    n, e = graph.num_nodes(), graph.num_edges()
    if not n or e % 2 or n + e // 2 != len(labels):
        raise ValueError(f"Graph/label dimensions disagree: {name}")
    src, dst = graph.edges(order="eid")
    if not torch.equal(src[::2], dst[1::2]) or not torch.equal(dst[::2], src[1::2]):
        raise ValueError(f"Directed edges are not consecutive reverse pairs: {name}")
    required = ((graph.ndata, n, {"x": (4, 7), "area": (2, 1), "ftype": (2, 11),
                                 "centroid": (2, 3), "rational": (2, 1)}),
                (graph.edata, e, {"x": (3, 6), "etype": (2, 9), "length": (2, 1),
                                 "convexity": (2, 3)}))
    for fields, size, spec in required:
        for key, (ndim, channels) in spec.items():
            value = fields[key]
            if value.ndim != ndim or value.shape[0] != size or value.shape[-1] != channels:
                raise ValueError(f"Unexpected feature shape for {name}/{key}: {tuple(value.shape)}")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Non-finite features: {name}/{key}")
    if data["face_labels"] + data["merged_edge_labels"] != labels:
        raise ValueError(f"Separate labels disagree with combined labels: {name}")
    if len(data["face_labels"]) != n or len(data["merged_edge_labels"]) != e // 2:
        raise ValueError(f"Separate label dimensions disagree: {name}")
    if data["num_undirected_edges"] != e // 2:
        raise ValueError(f"Unexpected undirected edge count: {name}")
    expected_mapping = [i // 2 for i in range(e)]
    if data["edge_to_undirected"] != expected_mapping:
        raise ValueError(f"Merged edge groups differ from graph edge pairs: {name}")
    if graph.edata["edge_to_undirected"].tolist() != expected_mapping:
        raise ValueError(f"Graph edge map disagrees with labels: {name}")
    return {key: data[key] for key in ("query", "labels", "face_labels", "merged_edge_labels",
                                      "edge_to_undirected", "num_undirected_edges")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/grounding_cache"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/grounding.json"))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    splits = load_split(args.split)
    count = 0
    for ids in splits.values():
        for name in ids:
            validate_state(args.dataset, name)
            count += 1
            if count % 2000 == 0:
                print(f"Checked {count} states", flush=True)
    print("Dataset structure and graph/label consistency checks passed.")


if __name__ == "__main__":
    main()
