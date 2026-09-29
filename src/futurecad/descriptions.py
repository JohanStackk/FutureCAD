import json
from pathlib import Path


def load_descriptions(ids, split, level="detail", local_path=None):
    if level not in {"overall", "detail"}:
        raise ValueError("Description level must be overall or detail")
    if local_path:
        descriptions = json.loads(Path(local_path).read_text(encoding="utf-8"))
        if not isinstance(descriptions, dict):
            raise ValueError("Descriptions must map CAD IDs to strings")
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download

        repo = "jhlee11/FutureCAD"
        split = "validation" if split == "val" else split
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported dataset split: {split}")
        api = HfApi()
        revision = api.dataset_info(repo).sha
        files = sorted(
            name
            for name in api.list_repo_files(
                repo, repo_type="dataset", revision=revision
            )
            if name.startswith(f"data/{split}-") and name.endswith(".parquet")
        )
        if not files:
            raise FileNotFoundError(f"No Parquet shards for {repo}/{split}")
        column = f"{level}_description"
        selected = set(ids)
        descriptions = {}
        for name in files:
            path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
            for batch in pq.ParquetFile(path).iter_batches(columns=["id", column]):
                for row in batch.to_pylist():
                    data_id = row["id"]
                    if data_id in selected:
                        if data_id in descriptions:
                            raise ValueError(f"Duplicate description ID: {data_id}")
                        descriptions[data_id] = row[column]
    invalid = [
        data_id
        for data_id in ids
        if not isinstance(descriptions.get(data_id), str)
        or not descriptions[data_id].strip()
    ]
    if not ids or invalid:
        raise ValueError(
            f"Missing or empty descriptions for selected IDs: {invalid[:10]}"
        )
    return {data_id: descriptions[data_id] for data_id in ids}
