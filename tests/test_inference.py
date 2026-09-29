import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cadquery as cq

from futurecad.export import export_predictions
from futurecad.text_to_cad import check_grounding_service, main


BOX = 'import cadquery as cq\nmodel = cq.Workplane("XY").box(1, 2, 3)'


class InferenceTests(unittest.TestCase):
    def test_export_step_roundtrip_and_failure_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cad"
            report = export_predictions(
                {"box": [BOX, "raise ValueError('bad CAD')"]}, output
            )
            self.assertEqual((report["total"], report["valid"]), (2, 1))
            self.assertIn("bad CAD", report["records"][1]["error"])
            folder = output / "box/sample0"
            self.assertGreater((folder / "model.stl").stat().st_size, 0)
            solid = cq.importers.importStep(str(folder / "model.step")).val()
            self.assertTrue(solid.isValid())
            self.assertAlmostEqual(solid.Volume(), 6)
            self.assertEqual(json.loads((output / "exports.json").read_text()), report)

    def test_export_rejects_unsafe_id_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_predictions({"../escape": [BOX]}, Path(tmp) / "out")
            with self.assertRaises(FileExistsError):
                export_predictions({"box": [BOX]}, tmp)

    def test_grounding_service_must_declare_release_threshold(self):
        for threshold in (0.5, 0.925):
            response = io.BytesIO(
                json.dumps(
                    {"status": "ready", "inference_threshold": threshold}
                ).encode()
            )
            with patch("futurecad.text_to_cad.urlopen", return_value=response):
                if threshold == 0.5:
                    with self.assertRaises(ValueError):
                        check_grounding_service("http://localhost:5678")
                else:
                    self.assertEqual(
                        check_grounding_service("http://localhost:5678")[
                            "inference_threshold"
                        ],
                        threshold,
                    )

    def test_orchestration_with_stub_generation_and_real_cad_export(self):
        real_run = subprocess.run
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"

            def run(command, **kwargs):
                if command[2] == "futurecad.generate":
                    split = json.loads(
                        Path(command[command.index("--split-file") + 1]).read_text()
                    )
                    self.assertEqual(split, {"train": [], "val": [], "test": ["demo"]})
                    self.assertEqual(command[command.index("--n") + 1], "1")
                    path = Path(command[command.index("--output") + 1])
                    path.write_text(json.dumps({"demo": [BOX]}))
                    return subprocess.CompletedProcess(command, 0)
                return real_run(command, **kwargs)

            with (
                patch(
                    "sys.argv",
                    ["text_to_cad", "--text", "A box", "--output-dir", str(output)],
                ),
                patch(
                    "futurecad.text_to_cad.check_grounding_service",
                    return_value={"status": "ready", "inference_threshold": 0.925},
                ),
                patch("futurecad.text_to_cad.subprocess.run", side_effect=run),
            ):
                main()
            self.assertEqual(
                json.loads((output / "run.json").read_text())["status"], "completed"
            )
            self.assertTrue((output / "cad/demo/sample0/model.step").is_file())

    def test_export_cli_returns_failure_for_invalid_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "input.json"
            predictions.write_text(json.dumps({"bad": ["raise ValueError('bad CAD')"]}))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "futurecad.export",
                    "--predictions",
                    str(predictions),
                    "--output-dir",
                    str(root / "out"),
                ],
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(
                json.loads((root / "out/exports.json").read_text())["valid"], 0
            )


if __name__ == "__main__":
    unittest.main()
