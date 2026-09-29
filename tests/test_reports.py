import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/report_metrics.py"
spec = importlib.util.spec_from_file_location("report_metrics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportTests(unittest.TestCase):
    def test_grounding_units_preserve_threshold_and_empty_rankings(self):
        rows = module.paper_rows(
            {
                "split": "test",
                "samples": 2,
                "inference_threshold": 0.925,
                "metrics": {
                    "f1": 0.53,
                    "mAP": None,
                    "recall@3": None,
                    "recall@5": None,
                    "recall@10": None,
                },
            }
        )
        self.assertEqual(rows[0]["f1"], 53)
        self.assertEqual(rows[0]["threshold"], 0.925)
        self.assertIsNone(rows[0]["mAP"])

    def test_llm_units_do_not_select_or_merge_samples(self):
        metrics = {
            "total": 2,
            "valid": 1,
            "G-F1": 0.8,
            "R-F1": 0.9,
            "IR": 0.5,
            "med-CD": 0.031,
            "avg-CD": 0.071,
            "f1_included": {"refine_f1": 1},
        }
        rows = module.paper_rows(
            {
                "reports": {
                    "sample0": {"advanced_op_test_set": metrics},
                    "sample1": {"advanced_op_test_set": {**metrics, "med-CD": None}},
                }
            }
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["G-F1"], 80)
        self.assertEqual(rows[0]["med-CD"], 31)
        self.assertIsNone(rows[1]["med-CD"])
        self.assertEqual(rows[0]["f1_included"], {"refine_f1": 1})


if __name__ == "__main__":
    unittest.main()
