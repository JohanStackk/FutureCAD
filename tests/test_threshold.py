import unittest

from futurecad.grounding_metrics import GroundingMetrics
from futurecad.grounding_threshold import checkpoint_threshold, resolve_threshold


class ThresholdTests(unittest.TestCase):
    def test_inference_rejects_training_threshold(self):
        with self.assertRaisesRegex(
            ValueError, "Checkpoint requires inference threshold 0.925"
        ):
            resolve_threshold(0.5, 0.925)
        self.assertEqual(resolve_threshold(0.925, 0.925), 0.925)
        self.assertEqual(resolve_threshold(0.5, 0.925, allow_override=True), 0.5)

    def test_embedded_threshold_validation(self):
        self.assertEqual(checkpoint_threshold("0.925"), 0.925)
        self.assertIsNone(checkpoint_threshold())
        with self.assertRaises(ValueError):
            checkpoint_threshold("nan")

    def test_threshold_changes_decisions_but_not_ranking(self):
        results = []
        for threshold in (0.5, 0.925):
            metrics = GroundingMetrics(threshold)
            metrics.update([[0.96, 0.8, 0.1, -1]], [[1, 0, 0, -1]])
            results.append(metrics.result())
        self.assertEqual(results[0]["fp"], 1)
        self.assertEqual(results[1]["fp"], 0)
        self.assertEqual(results[1]["f1"], 1)
        for key in ("mAP", "recall@3", "recall@5", "recall@10"):
            self.assertEqual(results[0][key], results[1][key])

    def test_invalid_thresholds(self):
        for threshold in (float("nan"), float("inf"), 0, 1, -0.1):
            with self.assertRaises(ValueError):
                resolve_threshold(threshold)


if __name__ == "__main__":
    unittest.main()
