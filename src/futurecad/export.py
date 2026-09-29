import argparse
import json
from pathlib import Path
import re
import subprocess

from futurecad.execution import execute_to_files


def export_predictions(predictions, output_dir, timeout=45):
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError(
            "Predictions must map IDs to nonempty lists of Python programs"
        )
    for data_id, codes in predictions.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", data_id):
            raise ValueError(f"Unsafe output ID: {data_id!r}")
        if (
            not isinstance(codes, list)
            or not codes
            or any(not isinstance(c, str) for c in codes)
        ):
            raise ValueError(f"Expected a nonempty program list for {data_id}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for data_id, codes in predictions.items():
        for sample, code in enumerate(codes):
            folder = output_dir / data_id / f"sample{sample}"
            folder.mkdir(parents=True)
            (folder / "model.py").write_text(code, encoding="utf-8")
            record = {
                "id": data_id,
                "sample": sample,
                "valid": False,
                "directory": folder.relative_to(output_dir).as_posix(),
            }
            try:
                execute_to_files(
                    code,
                    {"STEP": folder / "model.step", "STL": folder / "model.stl"},
                    timeout,
                )
                record["valid"] = True
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                record["error"] = str(error)[-2000:]
            records.append(record)
    report = {
        "total": len(records),
        "valid": sum(r["valid"] for r in records),
        "records": records,
    }
    (output_dir / "exports.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description='Execute generated programs and export STEP/STL files with execution records.')
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    report = export_predictions(predictions, args.output_dir, args.timeout)
    print(json.dumps(report, indent=2))
    if report["valid"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
