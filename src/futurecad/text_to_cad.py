import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

from futurecad.grounding_threshold import resolve_threshold


def check_grounding_service(url, threshold=0.925):
    with urlopen(url.rstrip("/") + "/health", timeout=10) as response:
        health = json.load(response)
    if health.get("status") != "ready" or "inference_threshold" not in health:
        raise RuntimeError(
            "Start the released BRepGround service with its inference config first"
        )
    resolve_threshold(threshold, health["inference_threshold"])
    return health


def main():
    parser = argparse.ArgumentParser(description='Generate one CAD program from text, then execute it to STEP and STL.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    parser.add_argument("--model", default="weights/futurecad-llm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-python", default=sys.executable)
    parser.add_argument(
        "--cad-python", default=os.environ.get("FUTURECAD_CAD_PYTHON", sys.executable)
    )
    parser.add_argument(
        "--grounding-url",
        default=os.environ.get("FUTURECAD_GROUNDING_URL", "http://127.0.0.1:5678"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    description = (
        args.text
        if args.text is not None
        else Path(args.text_file).read_text(encoding="utf-8")
    )
    if not description.strip() or args.timeout <= 0:
        parser.error("Description must be nonempty and timeout positive")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    health = check_grounding_service(args.grounding_url)
    output.mkdir(parents=True)
    descriptions, split, predictions = (
        output / "descriptions.json",
        output / "split.json",
        output / "predictions.json",
    )
    descriptions.write_text(json.dumps({"demo": description}), encoding="utf-8")
    split.write_text(
        json.dumps({"train": [], "val": [], "test": ["demo"]}), encoding="utf-8"
    )
    env = os.environ.copy()
    env["FUTURECAD_GROUNDING_URL"] = args.grounding_url
    env["FUTURECAD_CAD_PYTHON"] = args.cad_python
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    run = {"configuration": vars(args), "grounding": health, "status": "generating"}
    manifest = output / "run.json"

    def save():
        manifest.write_text(json.dumps(run, indent=2), encoding="utf-8")

    save()
    try:
        subprocess.run(
            [
                args.llm_python,
                "-m",
                "futurecad.generate",
                "--model",
                str(Path(args.model).resolve()),
                "--descriptions",
                str(descriptions),
                "--split-file",
                str(split),
                "--output",
                str(predictions),
                "--n",
                "1",
                "--seed",
                str(args.seed),
                "--temperature",
                str(args.temperature),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-model-len",
                str(args.max_model_len),
                "--max-tokens",
                str(args.max_tokens),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--enforce-eager",
            ],
            check=True,
            env=env,
        )
        run["status"] = "exporting"
        save()
        subprocess.run(
            [
                args.cad_python,
                "-m",
                "futurecad.export",
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output / "cad"),
                "--timeout",
                str(args.timeout),
            ],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as error:
        run["status"] = "failed"
        run["error"] = str(error)
        save()
        raise
    run["status"] = "completed"
    save()
    print(f"CAD files: {output / 'cad/demo/sample0'}")


if __name__ == "__main__":
    main()
