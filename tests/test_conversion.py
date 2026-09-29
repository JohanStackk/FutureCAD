import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from futurecad.history_to_code import convert_history


def box_history(refinement=None):
    points = [(-10, -5), (10, -5), (10, 5), (-10, 5)]
    curves = [
        {
            "type": "Line2D",
            "start_point": list(points[i]),
            "end_point": list(points[(i + 1) % 4]),
            "start_point_id": str(i),
            "end_point_id": str((i + 1) % 4),
        }
        for i in range(4)
    ]
    curves[1]["start_point"], curves[1]["end_point"] = (
        curves[1]["end_point"],
        curves[1]["start_point"],
    )
    curves[1]["start_point_id"], curves[1]["end_point_id"] = (
        curves[1]["end_point_id"],
        curves[1]["start_point_id"],
    )
    history = {
        "features": {
            "sketch": {
                "type": "sketch",
                "plane": {"origin": [0, 0, 0], "x": [1, 0, 0], "normal": [0, 0, 1]},
                "profiles": {"profile": {"loops": [{"loop_curves": curves}]}},
            },
            "extrude": {
                "type": "extrude",
                "sketch_id": "sketch",
                "parameters": {"depthOne": 6, "depthTwo": 0, "operationType": "NEW"},
            },
        },
        "sequence": [
            {"type": "sketch", "feature_id": "sketch"},
            {"type": "extrude", "feature_id": "extrude"},
        ],
    }
    if refinement:
        history["features"]["refine"] = {
            "type": refinement,
            "parameters": {
                {"fillet": "radius", "chamfer": "width", "shell": "thickness"}[
                    refinement
                ]: 1
            },
            "CoM": {
                "edges": []
                if refinement == "shell"
                else [[-10, 0, 6], [10, 0, 6], [0, -5, 6], [0, 5, 6]],
                "faces": [[0, 0, 6]] if refinement == "shell" else [],
            },
            "query": [
                'Select the "top" primitives.\nKeep \\ coordinates.',
                "Top boundary.",
                "Upper boundary.",
            ],
        }
        history["sequence"].append({"type": refinement, "feature_id": "refine"})
    return history


class ConversionTests(unittest.TestCase):
    def test_queries_escape_text_and_leave_history_unchanged(self):
        history = box_history("fillet")
        before = copy.deepcopy(history)
        convert_history(history, "points")
        codes = convert_history(history, "query")
        self.assertEqual(history, before)
        self.assertEqual(len(codes), 3)
        for index, code in enumerate(codes):
            calls = [
                n
                for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "Query"
            ]
            self.assertEqual(
                calls[0].args[0].value, history["features"]["refine"]["query"][index]
            )
        self.assertEqual(convert_history(history, "query", query_index=1), [codes[1]])

    def test_missing_query_does_not_drop_refinement(self):
        history = box_history("fillet")
        del history["features"]["refine"]["query"]
        with self.assertRaisesRegex(ValueError, "nonempty text queries"):
            convert_history(history, "query")
        with self.assertRaises(ValueError):
            convert_history(box_history(), "points", query_index=0)
        self.assertEqual(len(convert_history(box_history(), "query")), 1)

    def test_point_and_query_geometry_match_with_known_selection(self):
        import cadquery as cq
        import cq_monkey_patch
        from cq_monkey_patch.workplane_ext import WorkplaneEx

        cq_monkey_patch.set_filter_in_refine(False)
        try:
            for operation in ("fillet", "chamfer", "shell"):
                with self.subTest(operation=operation):
                    history = box_history(operation)
                    point_scope = {}
                    exec(convert_history(history, "points")[0], point_scope)
                    point_shape = point_scope["model"].val()
                    self.assertTrue(point_shape.isValid())
                    self.assertGreater(point_shape.Volume(), 0)
                    self.assertLess(point_shape.Volume(), 1200)

                    def selected_boxes(query, wrapped, operation=operation):
                        solid = cq.Shape.cast(wrapped)
                        if operation == "shell":
                            faces = [
                                f for f in solid.Faces() if abs(f.Center().z - 6) < 1e-6
                            ]
                            return [WorkplaneEx.get_bbox(f) for f in faces], []
                        edges = [
                            e for e in solid.Edges() if abs(e.Center().z - 6) < 1e-6
                        ]
                        return [], [WorkplaneEx.get_bbox(e) for e in edges]

                    with patch.object(cq.Query, "query", selected_boxes):
                        query_scope = {}
                        exec(convert_history(history, "query")[0], query_scope)
                    query_shape = query_scope["model"].val()
                    self.assertTrue(query_shape.isValid())
                    self.assertAlmostEqual(
                        query_shape.Volume(), point_shape.Volume(), places=7
                    )
                    self.assertEqual(len(query_shape.Faces()), len(point_shape.Faces()))
        finally:
            cq_monkey_patch.set_filter_in_refine(True)

    def test_cli_local_json_and_training_pairs(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = box_history("fillet")
            source = root / "00000001.json"
            source.write_text(json.dumps(history), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "futurecad.convert",
                    "--input",
                    str(source),
                    "--selection",
                    "both",
                    "--output-dir",
                    str(root / "local"),
                ],
                check=True,
                capture_output=True,
            )
            self.assertTrue((root / "local/points/00000001.py").is_file())
            self.assertEqual(len(list((root / "local/query").glob("*/*.py"))), 3)
            dataset = root / "dataset/data"
            dataset.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "id": "00000001",
                            "history": json.dumps(history),
                            "overall_description": "A rounded box.",
                            "detail_description": "A 20 by 10 by 6 box with a top fillet.",
                        }
                    ]
                ),
                dataset / "train-00000-of-00001.parquet",
            )
            args = [
                sys.executable,
                "-m",
                "futurecad.convert",
                "--dataset",
                str(dataset.parent),
                "--split",
                "train",
                "--selection",
                "query",
                "--training-pairs",
                "--output-dir",
                str(root / "training"),
            ]
            subprocess.run(args, check=True, capture_output=True)
            rows = [
                json.loads(line)
                for line in (root / "training/query.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {r["description_level"] for r in rows}, {"overall", "detail"}
            )
            for row in rows:
                code = (
                    root / "training/query" / str(row["query_index"]) / "00000001.py"
                ).read_text(encoding="utf-8")
                self.assertEqual(
                    row["messages"][1]["content"], f"```python\n{code}\n```"
                )
            self.assertNotEqual(
                subprocess.run(args, capture_output=True, check=False).returncode, 0
            )

    def test_query_variants_use_common_indices(self):
        history = box_history("fillet")
        extra = copy.deepcopy(history["features"]["refine"])
        extra["query"].append("Additional query.")
        history["features"]["second"] = extra
        history["sequence"].append({"type": "fillet", "feature_id": "second"})
        self.assertEqual(len(convert_history(history, "query")), 3)
        extra["query"] = ["One query."]
        self.assertEqual(len(convert_history(history, "query")), 1)


if __name__ == "__main__":
    unittest.main()
