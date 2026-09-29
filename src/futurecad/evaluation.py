import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from futurecad.code_metrics import program_f1, GROUPS
from futurecad.execution import execute_to_stl
from futurecad.geometry import (
    chamfer_dist,
    lower_median,
    normalize_pc,
    read_ply,
    stl2pc,
    write_ply,
)
from futurecad.splits import split_ids


def evaluate_program(code, gt_pc, counts, seed, timeout):
    result = {"valid": False, "cd": None, "f1": None}
    try:
        with tempfile.TemporaryDirectory(prefix="futurecad-eval-") as tmp:
            stl, ply = Path(tmp) / "model.stl", Path(tmp) / "model.ply"
            execute_to_stl(code, stl, timeout=timeout)
            write_ply(normalize_pc(stl2pc(stl, 2048, seed=seed)), ply)
            result["cd"] = chamfer_dist(read_ply(ply), gt_pc)
        result["valid"] = True
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)[-500:]
        return result
    try:
        result["f1"] = program_f1(code, counts)
    except SyntaxError as error:
        result["f1_error"] = str(error)
    return result


def summarize(records):
    valid = [r for r in records if r["valid"]]
    distances = [r["cd"] for r in valid]
    groups = {
        key: [r["f1"][key] for r in valid if r["f1"] is not None and r["f1"][key] > 0]
        for key in GROUPS
    }
    f1 = {
        key: float(np.mean(values)) if values else 0.0 for key, values in groups.items()
    }
    return {
        "total": len(records),
        "valid": len(valid),
        "IR": 1 - len(valid) / len(records) if records else None,
        "med-CD": lower_median(distances) if distances else None,
        "avg-CD": float(np.mean(distances)) if distances else None,
        **f1,
        "G-F1": (f1["curve_f1"] + f1["3d_op_f1"]) / 2,
        "R-F1": f1["refine_f1"],
        "f1_included": {key: len(values) for key, values in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser('Evaluate generated ID -> [CAD program] JSON against an explicit test split.')
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--gt-ply-dir", required=True, help="Already-normalized ground-truth PLYs"
    )
    parser.add_argument("--gt-counts", required=True)
    parser.add_argument("--advanced-ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.workers < 1 or args.timeout <= 0:
        raise ValueError("Workers and timeout must be positive")
    ids = split_ids(args.split_file, args.split)
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    counts = json.loads(Path(args.gt_counts).read_text(encoding="utf-8"))
    advanced = (
        set(json.loads(Path(args.advanced_ids).read_text()))
        if args.advanced_ids
        else set()
    )
    if not ids or set(predictions) != set(ids):
        raise ValueError(
            "Prediction IDs must exactly equal the requested nonempty split"
        )
    n = len(predictions[ids[0]])
    if n < 1 or any(
        not isinstance(predictions[i], list)
        or len(predictions[i]) != n
        or any(not isinstance(code, str) for code in predictions[i])
        for i in ids
    ):
        raise ValueError(
            "Predictions must contain equally sized, nonempty lists of program strings"
        )


    for data_id in ids:
        read_ply(Path(args.gt_ply_dir) / f"{data_id}.ply")
        if data_id not in counts:
            raise KeyError(f"Missing command counts for {data_id}")
        for keys in GROUPS.values():
            for key in keys:
                if (
                    not isinstance(counts[data_id][key], (int, float))
                    or counts[data_id][key] < 0
                ):
                    raise ValueError(f"Invalid command count for {data_id}/{key}")

    def run(item):
        data_id, sample = item
        seed_bytes = hashlib.sha256(
            f"{args.seed}:{data_id}:{sample}".encode()
        ).digest()[:4]
        result = evaluate_program(
            predictions[data_id][sample],
            read_ply(Path(args.gt_ply_dir) / f"{data_id}.ply"),
            counts[data_id],
            int.from_bytes(seed_bytes, "little"),
            args.timeout,
        )
        return data_id, sample, result

    records = {i: [None] * n for i in ids}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for data_id, sample, result in pool.map(
            run, ((i, s) for i in ids for s in range(n))
        ):
            records[data_id][sample] = result
    reports = {}
    for sample in range(n):
        reports[f"sample{sample}"] = {
            "full_test_set": summarize([records[i][sample] for i in ids]),
            "non_advanced_op_test_set": summarize(
                [records[i][sample] for i in ids if i not in advanced]
            ),
            "advanced_op_test_set": summarize(
                [records[i][sample] for i in ids if i in advanced]
            ),
        }
    result = {
        "configuration": vars(args),
        "reports": reports,
        "per_example": records,
        "protocol": "operation-count-f1; bidirectional squared CD sum; lower median; seeded 2048 points",
        "units": "fractions; paper CD is multiplied by 1000; paper F1/IR by 100",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
