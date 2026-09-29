import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from futurecad.splits import load_splits, split_ids
from futurecad.geometry import (
    chamfer_dist,
    lower_median,
    normalize_pc,
    reward_linear_cd,
)
from futurecad.grounding_metrics import GroundingMetrics
from futurecad.code_metrics import program_f1, count_f1
from futurecad.evaluation import summarize
from futurecad.generate import extract_code
from futurecad.rewards import cad_reward as reward


class SplitTests(unittest.TestCase):
    def test_validation_alias_does_not_select_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            path.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]}))
            self.assertEqual(split_ids(path, "validation"), ["b"])
            with self.assertRaises(ValueError):
                split_ids(path, "validaton")
            path.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["b"]}))
            with self.assertRaises(ValueError):
                load_splits(path)

    def test_duplicate_and_conflicting_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            for obj in (
                {"train": ["a", "a"], "val": [], "test": []},
                {"train": [], "val": ["b"], "validation": ["c"], "test": []},
            ):
                path.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):
                    load_splits(path)


class GeometryTests(unittest.TestCase):
    def test_squared_chamfer_matches_bruteforce(self):
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(31, 3)), rng.normal(size=(25, 3))
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        expected = squared.min(axis=0).mean() + squared.min(axis=1).mean()
        self.assertAlmostEqual(chamfer_dist(a, b), expected, places=12)

    def test_normalization_and_degenerate_points(self):
        points = np.array([[1.0, 2, 3], [3.0, 2, 1], [0, 1, 2]])
        result = normalize_pc(points)
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(result, axis=1).max(), 1)
        for invalid in (np.zeros((4, 3)), np.empty((0, 3)), np.array([[np.nan, 0, 1]])):
            with self.assertRaises(ValueError):
                normalize_pc(invalid)

    def test_lower_median_and_reward_boundary(self):
        self.assertEqual(lower_median([9, 1, 2, 3]), 2)
        self.assertEqual(reward_linear_cd(0), 1)
        self.assertAlmostEqual(reward_linear_cd(0.1), 0.5)
        self.assertEqual(reward_linear_cd(0.2), 0)
        self.assertEqual(reward_linear_cd(2), 0)


class GroundingTests(unittest.TestCase):
    def test_padding_micro_f1_and_macro_ranking(self):
        metrics = GroundingMetrics(ks=(1, 3))
        metrics.update(
            [[0.9, 0.8, 0.1, 0.99], [0.2, 0.9, 0.1, 0.0]],
            [[1, 0, 1, -1], [0, 1, -1, -1]],
        )
        result = metrics.result()
        self.assertEqual(
            (result["tp"], result["fp"], result["fn"], result["tn"]), (2, 1, 1, 1)
        )
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["mAP"], ((1 + 2 / 3) / 2 + 1) / 2)
        self.assertAlmostEqual(result["recall@1"], 0.75)
        self.assertEqual(result["recall@3"], 1)

    def test_no_positive_query_does_not_change_ap_denominator(self):
        metrics = GroundingMetrics()
        metrics.update([[0.9, 0.1]], [[0, 0]])
        self.assertIsNone(metrics.result()["mAP"])
        self.assertEqual(metrics.result()["fp"], 1)


class RewardTests(unittest.TestCase):
    def test_missing_fence_does_not_touch_ground_truth(self):
        self.assertEqual(reward.compute_score(None, "plain text", {}), 0)

    @patch.object(reward, "read_ply", return_value=np.array([[0.0, 0, 0], [1, 0, 0]]))
    @patch.object(reward, "execute_to_stl", side_effect=RuntimeError("invalid CAD"))
    def test_invalid_execution_scores_zero(self, execute, read):
        self.assertEqual(
            reward.compute_score(None, "```python\nbad\n```", {"ply_path": "gt"}), 0
        )

    def test_success_formula_and_last_code_block(self):
        points = np.array([[-1.0, 0, 0], [1, 0, 0]])
        with (
            patch.object(reward, "read_ply", return_value=points),
            patch.object(reward, "execute_to_stl") as execute,
            patch.object(reward, "stl2pc", return_value=points),
            patch.object(reward, "write_ply"),
            patch.object(reward, "chamfer_dist", return_value=0.1),
        ):
            score = reward.compute_score(
                None, "```python\nfirst\n```\n```python\nlast\n```", {"ply_path": "gt"}
            )
            self.assertAlmostEqual(score, 0.55)
            self.assertEqual(execute.call_args.args[0], "last")


class CodeMetricTests(unittest.TestCase):
    def test_evaluation_extracts_last_block_and_accepts_unfenced_code(self):
        self.assertEqual(
            extract_code("```python\nfirst\n```\n```python\nlast\n```"), "last"
        )
        self.assertEqual(extract_code("model=1"), "model=1")

    def test_exclusions_and_count_groups(self):
        gt = {
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
        gt.update(circle=1, extrude=1, fillet=2)
        self.assertEqual(
            program_f1("model=cq.circle(1).extrude(1).fillet(1)", gt),
            {"curve_f1": 1.0, "3d_op_f1": 1.0, "refine_f1": 2 / 3},
        )
        self.assertIsNone(program_f1("model=cq.mirror()", gt))
        self.assertIsNone(program_f1("for x in []:\n pass", gt))
        self.assertEqual(count_f1(2, 0), -1)

    def test_failures_in_invalid_ratio_not_f1(self):
        result = summarize(
            [
                {"valid": False},
                {
                    "valid": True,
                    "cd": 0.01,
                    "f1": {"curve_f1": 1.0, "3d_op_f1": 0.5, "refine_f1": -1.0},
                },
            ]
        )
        self.assertEqual(result["IR"], 0.5)
        self.assertEqual(result["G-F1"], 0.75)
        self.assertEqual(result["f1_included"]["refine_f1"], 0)
        self.assertIsNone(summarize([])["IR"])


if __name__ == "__main__":
    unittest.main()
