import json
from pathlib import Path


def load_splits(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "val" in raw and "validation" in raw and raw["val"] != raw["validation"]:
        raise ValueError("Conflicting val and validation lists")
    result = {}
    for split in ("train", "val", "test"):
        key = "validation" if split == "val" and "val" not in raw else split
        ids = raw[key]
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise ValueError(f"{key} must be a list of string IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs within {key}")
        result[split] = ids
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(result[a]) & set(result[b])
        if overlap:
            raise ValueError(f"{a}/{b} overlap: {len(overlap)} IDs")
    return result


def split_ids(path, split):
    split = {"validation": "val"}.get(split, split)
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split!r}")
    return load_splits(path)[split]
