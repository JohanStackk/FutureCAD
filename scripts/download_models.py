import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description='Download FutureCAD and BRepGround models from Hugging Face.')
    parser.add_argument("--model", choices=("all", "grounding", "llm"), default="all")
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()
    models = {
        "grounding": ("jhlee11/BRepGround", "brepground"),
        "llm": ("jhlee11/FutureCAD-7B", "futurecad-llm"),
    }
    for name, (repo, directory) in models.items():
        if args.model in {"all", name}:
            snapshot_download(repo_id=repo, local_dir=args.output / directory)


if __name__ == "__main__":
    main()
