from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from futurecad.execution import execute_to_stl
from futurecad.geometry import stl2pc, write_ply, read_ply


class CADTests(unittest.TestCase):
    def test_box_export_seeded_sampling_and_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            stl, ply = Path(tmp) / "box.stl", Path(tmp) / "box.ply"
            execute_to_stl(
                'import cadquery as cq\nmodel=cq.Workplane("XY").box(1,2,3)', stl
            )
            a, b = stl2pc(stl, 2048, seed=42), stl2pc(stl, 2048, seed=42)
            np.testing.assert_array_equal(a, b)
            write_ply(a, ply)
            points = read_ply(ply)
            self.assertEqual(points.shape, (2048, 3))
            self.assertAlmostEqual(np.linalg.norm(points, axis=1).max(), 1, places=6)

    def test_invalid_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                execute_to_stl('raise ValueError("invalid")', Path(tmp) / "bad.stl")
            with self.assertRaises(subprocess.TimeoutExpired):
                execute_to_stl("while True: pass", Path(tmp) / "hang.stl", timeout=0.5)

    def test_evaluation_cli_counts_failed_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = (
                'import cadquery as cq\nmodel=cq.Workplane("XY").circle(1).extrude(2)'
            )
            stl = root / "reference.stl"
            execute_to_stl(code, stl)
            points = stl2pc(stl, 2048, seed=11)
            for name in ("valid", "invalid"):
                write_ply(points, root / f"{name}.ply")
            counts = {
                key: 0
                for key in (
                    "line",
                    "arc",
                    "circle",
                    "spline",
                    "extrude",
                    "revolve",
                    "fillet",
                    "chamfer",
                    "shell",
                )
            }
            counts.update(circle=1, extrude=1)
            inputs = {
                "split.json": {"train": [], "val": [], "test": ["valid", "invalid"]},
                "predictions.json": {
                    "valid": [code],
                    "invalid": ['raise ValueError("bad model")'],
                },
                "counts.json": {"valid": counts, "invalid": counts},
            }
            for name, data in inputs.items():
                (root / name).write_text(json.dumps(data))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "futurecad.evaluation",
                    "--predictions",
                    str(root / "predictions.json"),
                    "--split-file",
                    str(root / "split.json"),
                    "--gt-ply-dir",
                    str(root),
                    "--gt-counts",
                    str(root / "counts.json"),
                    "--output",
                    str(root / "report.json"),
                    "--workers",
                    "1",
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            report = json.loads((root / "report.json").read_text())["reports"][
                "sample0"
            ]["full_test_set"]
            self.assertEqual(report["IR"], 0.5)
            self.assertEqual(report["valid"], 1)
            self.assertEqual(report["G-F1"], 1)


if __name__ == "__main__":
    unittest.main()
