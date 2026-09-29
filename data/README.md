# FutureCAD data

This directory contains FutureCAD splits and evaluation references.

The BRepGround download includes input STEP files, text queries, and binary
labels for every OCC face and edge. See
[the data format and label-reading rules](../docs/dataset.md) for the
pythonocc environment, traversal convention and runnable examples.

| File | Purpose | Entries |
|---|---|---|
| `splits/futurecad.json` | FutureCAD program train/val/test IDs | 129,020 / 7,167 / 7,168 |
| `splits/grounding.json` | BRepGround state train/val/test IDs | 51,034 / 2,837 / 2,857 |
| `splits/advanced_ids.json` | FutureCAD IDs with advanced operations | 28,100 |
| `evaluation/futurecad_operation_counts.json` | Ground-truth counts for FutureCAD validation/test | 14,335 |

## Splits

Grounding states expand into 8,511 validation and 8,575 test queries with the
released labels. Advanced train/val/test intersections are 25,294 / 1,407 / 1,399.
Exact IDs are disjoint across splits, but grounding parent CAD models overlap:
train/val 2,039, train/test 2,032, val/test 272. These are not family-disjoint splits.

Grounding configs use `data/splits/grounding.json`. Set `data.limit` to run
a smoke check on a smaller number of states.

## Operation-count references

`futurecad_operation_counts.json` maps validation/test CAD IDs to ground-truth
counts of `line`, `arc`, `circle`, `spline`, `extrude`, `revolve`, `fillet`,
`chamfer`, and `shell` for G-F1/R-F1.

The evaluation script uses `futurecad_operation_counts.json` together with the
FutureCAD split and Advanced subset IDs. CD requires normalized `<id>.ply` files in addition to
these counts. See [dataset usage](../docs/dataset.md).

## Validation

```bash
python scripts/validate_data.py --data-dir data
```

The checker validates split disjointness, Advanced membership, and evaluation
reference coverage.

## Hugging Face datasets

[FutureCAD](https://huggingface.co/datasets/jhlee11/FutureCAD) contains the modeling
histories and `overall_description` / `detail_description` in split-specific
Parquet files. The `history` column is a JSON string;
use `json.loads` to restore the original `features` and `sequence` dictionary.
The evaluation script reads detailed descriptions directly from Hugging Face.
For custom pipelines, `export_data.py --format descriptions` optionally exports
an ID-to-text JSON.

For LLM training, pair each description with the complete CadQuery program
converted from the history with the same ID. See
[training target formats](../docs/dataset.md#text-to-cad-training-targets) for
point selection and the paper's BRep primitive queries, including their
execution requirements. `python -m futurecad.convert --split train --selection query
--training-pairs --output-dir outputs/training-query` downloads the training
histories and descriptions and writes the programs and paired chat records.

[BRepGround](https://huggingface.co/datasets/jhlee11/BRepGround) contains text and
labels in Parquet and STEP/label files in `assets/`. Its `export_data.py`
restores `datasets/grounding`. For BRepGround training and evaluation,
`scripts/prepare_grounding.py` builds `datasets/grounding_cache` locally.
