import argparse
import json
import re
from pathlib import Path

from futurecad.history_to_code import convert_history
from futurecad.prompts import prompt_sft


def dataset_rows(source, split, sample_id=None):
    import pyarrow.parquet as pq

    split = "validation" if split == "val" else split
    local = Path(source).is_dir()
    if local:
        files = sorted(Path(source).glob(f"data/{split}-*.parquet"))
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        revision = api.dataset_info(source).sha
        names = sorted(
            name
            for name in api.list_repo_files(
                source, repo_type="dataset", revision=revision
            )
            if name.startswith(f"data/{split}-") and name.endswith(".parquet")
        )
        files = (
            hf_hub_download(source, name, repo_type="dataset", revision=revision)
            for name in names
        )
    found = False
    seen = set()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=128,
            columns=["id", "history", "overall_description", "detail_description"],
        ):
            for row in batch.to_pylist():
                if sample_id is not None and row["id"] != sample_id:
                    continue
                if row["id"] in seen:
                    raise ValueError(f"Duplicate CAD ID: {row['id']}")
                seen.add(row["id"])
                found = True
                yield row
    if not found:
        raise ValueError(f"No matching samples in {source}/{split}: {sample_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FutureCAD histories into CadQuery programs."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="One modeling-history JSON file")
    source.add_argument(
        "--dataset",
        default="jhlee11/FutureCAD",
        help="Hub dataset ID or local Parquet dataset directory",
    )
    parser.add_argument(
        "--split", choices=["train", "validation", "val", "test"], default="test"
    )
    parser.add_argument("--id", dest="sample_id")
    parser.add_argument(
        "--selection", choices=["points", "query", "both"], default="points"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-pairs",
        action="store_true",
        help="Also write description/program chat pairs from dataset rows",
    )
    args = parser.parse_args()
    if args.input and (args.sample_id or args.training_pairs):
        parser.error("--id and --training-pairs require a dataset source")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    if args.input:
        rows = [
            {"id": args.input.stem, "history": args.input.read_text(encoding="utf-8")}
        ]
    else:
        rows = dataset_rows(args.dataset, args.split, args.sample_id)
    selections = ["points", "query"] if args.selection == "both" else [args.selection]
    handles = {}
    count = 0
    try:
        for row in rows:
            data_id = row["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", data_id):
                raise ValueError(f"Invalid CAD ID: {data_id!r}")
            history = json.loads(row["history"])
            for selection in selections:
                try:
                    programs = convert_history(history, selection=selection)
                except Exception as error:
                    raise ValueError(
                        f"Conversion failed for {data_id} ({selection}): {error}"
                    ) from error
                if args.training_pairs:
                    for level in ("overall", "detail"):
                        text = row[f"{level}_description"]
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f"Missing {level} description: {data_id}")
                for index, code in enumerate(programs):
                    directory = args.output_dir / selection
                    if selection == "query":
                        directory /= str(index)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"{data_id}.py").write_text(code, encoding="utf-8")
                    if args.training_pairs:
                        if selection not in handles:
                            handles[selection] = (
                                args.output_dir / f"{selection}.jsonl"
                            ).open("x", encoding="utf-8")
                        for level in ("overall", "detail"):
                            record = {
                                "id": data_id,
                                "description_level": level,
                                "query_index": index if selection == "query" else None,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": prompt_sft(
                                            row[f"{level}_description"]
                                        ),
                                    },
                                    {
                                        "role": "assistant",
                                        "content": f"```python\n{code}\n```",
                                    },
                                ],
                            }
                            handles[selection].write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                    count += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(f"Wrote {count} CadQuery programs to {args.output_dir}")


if __name__ == "__main__":
    main()
