import argparse
from pathlib import Path

from futurecad.execution import execute_to_files


def main():
    parser = argparse.ArgumentParser(description='Execute a CadQuery program and export its solid and isometric SVG view.')
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    code = args.program.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {kind: args.output_dir / f"model.{kind.lower()}" for kind in ("STEP", "STL", "SVG")}
    execute_to_files(code, outputs, timeout=args.timeout)
    print(f"CAD and SVG files: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
