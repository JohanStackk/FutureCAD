import os
from pathlib import Path
import subprocess
import sys
import tempfile


def execute_to_stl(code, output, timeout=45, strict=False):
    return execute_to_files(code, {"STL": output}, timeout=timeout, strict=strict)


def execute_to_files(code, outputs, timeout=45, strict=False):
    if not outputs or set(outputs) - {"STL", "STEP", "SVG"}:
        raise ValueError("Outputs must request STL, STEP or SVG")
    outputs = {kind: str(Path(path).resolve()) for kind, path in outputs.items()}
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Each export format needs a distinct output path")
    for path in outputs.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    patched = code.replace(
        "import cq_monkey_patch",
        "import cq_monkey_patch\ncq_monkey_patch.set_filter_in_refine("
        + str(not strict)
        + ")",
    )
    worker = (
        patched
        + "\n\nimport cadquery as _cq\n"
        + """
if "model" not in globals():
    raise RuntimeError("CAD program must define model")
if not model.val().isValid():
    raise RuntimeError("CAD kernel reports an invalid shape")
"""
        + "\n".join(
            f"_cq.exporters.export(model, {path!r}, exportType={kind!r})"
            for kind, path in outputs.items()
        )
    )
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="futurecad-") as tmp:
        script = Path(tmp) / "program.py"
        script.write_text(worker, encoding="utf-8")
        python = os.environ.get("FUTURECAD_CAD_PYTHON", sys.executable)
        process = subprocess.run(
            [python, str(script)],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if process.returncode:
            raise RuntimeError(
                f"CAD worker exited with code {process.returncode}: "
                + process.stderr[-2000:]
            )
    for path in outputs.values():
        if not Path(path).is_file() or not Path(path).stat().st_size:
            raise RuntimeError("CAD program did not export a nonempty model")
