import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalDataTests(unittest.TestCase):
    def test_released_data_integrity_and_reference_coverage(self):
        validator = load_module(
            "validate_release_data", ROOT / "scripts/validate_data.py"
        )
        report = validator.validate(ROOT / "data")
        self.assertEqual(report["futurecad_evaluation_count_coverage"], 14335)
        self.assertEqual(
            report["grounding_state_counts"],
            {"train": 51034, "val": 2837, "test": 2857},
        )
        self.assertFalse((ROOT / "data/splits/valid_ids.json").exists())

    def test_dataset_uses_explicit_final_split_without_hidden_filter(self):
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")
        data.Dataset = object
        data.DataLoader = object
        with patch.dict(
            "sys.modules",
            {"torch": torch, "torch.utils": utils, "torch.utils.data": data},
        ):
            module = load_module(
                "isolated_grounding_dataset", ROOT / "src/grounding_brep/dataset.py"
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "graph").mkdir()
            (root / "labels").mkdir()
            for name in ("00000001_1", "00000002_1"):
                (root / "graph" / f"{name}.bin").touch()
                (root / "labels" / f"{name}.json").write_text(
                    json.dumps({"query": ["a", "b"], "labels": [1]})
                )
            split = root / "final.json"
            split.write_text(
                json.dumps(
                    {"train": [], "val": [], "test": ["00000001_1", "00000002_1"]}
                )
            )
            (root / "splits.json").write_text(
                json.dumps({"train": [], "val": [], "test": ["missing"]})
            )
            with self.assertRaisesRegex(ValueError, "final split"):
                module.GroundingBRepDataset(root, split="test")
            dataset = module.GroundingBRepDataset(
                root, split="test", split_path=split, limit=1
            )
            self.assertEqual(dataset.samples, [("00000001_1", 0), ("00000001_1", 1)])
            self.assertEqual(
                len(module.GroundingBRepDataset(root, split="test", split_path=split)),
                4,
            )


if __name__ == "__main__":
    unittest.main()
