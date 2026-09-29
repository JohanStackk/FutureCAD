import argparse
import json
from pathlib import Path


def paper_rows(report):
    if "inference_threshold" in report and "metrics" in report:
        m = report["metrics"]
        return [
            {
                "split": report["split"],
                "queries": report["samples"],
                "threshold": report["inference_threshold"],
                **{
                    k: None if m[k] is None else 100 * m[k]
                    for k in ("f1", "mAP", "recall@3", "recall@5", "recall@10")
                },
            }
        ]
    return [
        {
            "sample": sample,
            "subset": subset,
            "total": m["total"],
            "valid": m["valid"],
            **{
                k: None if m[k] is None else m[k] * scale
                for k, scale in (
                    ("G-F1", 100),
                    ("R-F1", 100),
                    ("IR", 100),
                    ("med-CD", 1000),
                    ("avg-CD", 1000),
                )
            },
            "f1_included": m["f1_included"],
        }
        for sample, groups in report["reports"].items()
        for subset, m in groups.items()
    ]


def main():
    parser = argparse.ArgumentParser(description='Print evaluation metrics.')
    parser.add_argument("report")
    args = parser.parse_args()
    print(
        json.dumps(
            paper_rows(json.loads(Path(args.report).read_text(encoding="utf-8"))),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
