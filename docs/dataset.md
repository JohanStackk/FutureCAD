# Datasets

[FutureCAD](https://huggingface.co/datasets/jhlee11/FutureCAD) contains modeling
histories and two levels of text descriptions. [BRepGround](https://huggingface.co/datasets/jhlee11/BRepGround) contains
STEP solids, text queries, and face/edge labels. Both use CC BY 4.0.

## FutureCAD

Read a test Parquet shard using the libraries in the `futurecad` environment:

```python
import json
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

path = hf_hub_download(
    "jhlee11/FutureCAD",
    "data/test-00000-of-00002.parquet",
    repo_type="dataset",
)
sample = pq.read_table(path).slice(0, 1).to_pylist()[0]
history = json.loads(sample["history"])
features, sequence = history["features"], history["sequence"]
overall_description = sample["overall_description"]
detail_description = sample["detail_description"]
```

`history` preserves the modeling-history dictionary as a JSON string.
The other columns identify the model, its Advanced
subset membership, and its operations. Split IDs and operation-count references
are listed in [data/README.md](../data/README.md).

`overall_description` summarizes shape and structure; `detail_description` adds
geometry and scale. Both levels are used for SFT; the main evaluation uses the
detailed descriptions.

### Download histories and descriptions

Read histories and both description levels directly from the Parquet rows above.
If individual files are needed for a custom training pipeline, export them with:

```bash
hf download jhlee11/FutureCAD export_data.py --repo-type dataset --local-dir downloads/futurecad
python downloads/futurecad/export_data.py --split train --format history --output datasets/futurecad/train/history
python downloads/futurecad/export_data.py --split train --format descriptions --description-level overall --output datasets/futurecad/train/overall
python downloads/futurecad/export_data.py --split train --format descriptions --description-level detail --output datasets/futurecad/train/detail
```

Each `history/<id>.json` pairs with the entry keyed by the same CAD ID in
`overall/descriptions.json` and `detail/descriptions.json`. Keep IDs as strings,
including leading zeros. Replace `train` with `validation` or `test` in both
`--split` and the output paths to export another split.

### Convert JSON to CadQuery

Convert one sample directly from Hugging Face into both program formats:

```bash
python -m futurecad.convert --split test --id 00114630 --selection both --output-dir outputs/converted-example
python visualize/render.py --program outputs/converted-example/points/00114630.py --output-dir outputs/converted-view
```

The output contains `points/00114630.py` and
`query/0/00114630.py`, `query/1/00114630.py`, `query/2/00114630.py`.
To convert a local modeling-history JSON:

```bash
python -m futurecad.convert --input model.json --selection both --output-dir outputs/converted-local
```

Use `--selection points` or `--selection query` to generate one format. Use a
new output directory for each run. The converter preserves the input history
and its coordinate system. It supports sketches with lines, arcs, circles and
interpolated splines, extrusion, revolution, Boolean operations, fillet,
chamfer and shell. The implementation is
[history_to_code.py](../src/futurecad/history_to_code.py).

### Text-to-CAD training targets

The LLM input is a description of the complete CAD model. Its target is the
complete CadQuery Python program converted from that model's history JSON,
with the final object assigned to `model`. Each description level forms a
separate input paired with the program. The SFT prompt is defined in
[futurecad/prompts.py](../src/futurecad/prompts.py).

Generate the paper's query-format targets and description/program pairs from
the training split:

```bash
python -m futurecad.convert --split train --selection query --training-pairs --output-dir outputs/training-query
```

This writes the `.py` programs under `query/<query-index>/<id>.py` and chat
training records to `query.jsonl`. Each record contains `id`,
`description_level`, `query_index`, and `messages`: the user message is the SFT
prompt with the model description, and the assistant message is the complete
CadQuery program in a Python code block. Both description levels are paired
with each program. Use `--selection both` to also write `points/<id>.py` and
`points.jsonl`. For a local Parquet dataset copy, add `--dataset /path/to/FutureCAD`.

Query variants use the same query index at every refinement operation, with
up to three variants per model. When operations have different numbers of
queries, only indices available at every operation are used. Models without
refinements produce one program. Missing refinement queries raise an error.

For `fillet`, `chamfer`, and `shell`, the program has two selection formats:

| Program format | Primitive selection | Execution |
|---|---|---|
| Point selection | Coordinates select edges for fillet/chamfer and faces for shell. | Executes locally to validate the converted geometry; no BRepGround service is needed. |
| BRep primitive query | Natural-language queries select the primitives for `filletQ`, `chamferQ`, and `shellQ`. | The paper's program format; requires `cq_monkey_patch` and the BRepGround service. |

Point-selection programs using `cq.selectors.NearestToPointsSelector` also import
`cq_monkey_patch`, which registers this selector. It selects the edge or face
whose **center** is nearest to each supplied point. These points must refer to
the solid at that operation in the construction sequence.

For example, these two programs express a fillet on the top perimeter of a box.
They illustrate the selection formats, rather than a dataset sample.

Point selection:

```python
import cq_monkey_patch
import cadquery as cq

model = cq.Workplane("XY").box(20, 10, 6)
points = [(-10, 0, 3), (10, 0, 3), (0, -5, 3), (0, 5, 3)]
model = model.edges(cq.selectors.NearestToPointsSelector(points)).fillet(1)
```

BRep primitive query:

```python
import cq_monkey_patch
import cadquery as cq

model = cq.Workplane("XY").box(20, 10, 6)
model = model.filletQ(cq.Query("Select the four perimeter edges of the top face."), radius=1)
```

The corresponding chamfer and shell calls are
`model.chamferQ(cq.Query(edge_query), length=1)` and
`model.shellQ(cq.Query(face_query), thickness=-1)`. Their point-selection
counterparts use `.edges(selector).chamfer(1)` and `.faces(selector).shell(-1)`.
The text inside `cq.Query` describes primitives in the current intermediate
solid; it is distinct from the full-model input description.

Save a program as `model.py` and execute/render it with:

```bash
python visualize/render.py --program model.py --output-dir outputs/model-view
```

For the query format, first start `bash scripts/serve_grounding.sh` in another
terminal. BRepGround predicts the selected faces/edges, and the monkey-patch
applies the requested operation. Geometry validation with point selection is
independent of those predictions.

The Hugging Face dataset stores histories and descriptions; the converter
above generates the target programs locally.

## Rewritten programs

The `rewritten` configuration of [FutureCAD](https://huggingface.co/datasets/jhlee11/FutureCAD)
contains Claude-Sonnet-4.5 rewritten CadQuery programs and both description
levels. Each row uses the original eight-character CAD model ID and inherits
its split. The train, validation, and test splits contain 77,835, 4,335, and
4,345 programs, respectively.

Download and read the rewrite Parquet files:

```bash
hf download jhlee11/FutureCAD --repo-type dataset --include "rewritten/*.parquet" --local-dir datasets/futurecad
```

```python
from pathlib import Path
import pyarrow.dataset as ds

files = sorted(Path("datasets/futurecad/rewritten").glob("train-*.parquet"))
rewritten = ds.dataset([str(path) for path in files], format="parquet")
sample = next(rewritten.to_batches(batch_size=1)).to_pylist()[0]
code = sample["code"]
overall_description = sample["overall_description"]
detail_description = sample["detail_description"]
```

Each program pairs with both description levels. `id` matches the same model
in the dataset's default configuration; retain leading zeros. Programs and descriptions retain
their original text. Save `code` as a Python file and use `visualize/render.py`
to execute it and export geometry; start BRepGround first when the program uses
`cq.Query`.

## BRepGround

### OCC executability and selection semantics

OpenCascade (OCC) has limited robustness for fillet, chamfer, and shell on some
geometries. During dataset construction, referenced edges or faces for which
OCC could not execute the requested operation were filtered out. Consequently,
the released target selection can differ from the original Onshape user action;
for example, some originally selected edges may be absent. The released entity
labels and text queries refer to the retained, OCC-executable target selection.

Restore the STEP and label files:

```bash
hf download jhlee11/BRepGround export_data.py --repo-type dataset --local-dir downloads/brepground
python downloads/brepground/export_data.py --split all --output datasets/grounding
```

```text
datasets/grounding/
  steps/<state-id>.step
  labels/<state-id>.json
```

Label JSON contains `query` (a list of text queries sharing the target),
`face_labels` and `edge_labels` (binary arrays). Use **pythonocc-core 7.8.1.1 /
OCCT 7.8.1** and the following traversal:

```python
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

shape = read_step_file("model.step", as_compound=True, verbosity=False)
topology = TopologyExplorer(shape, ignore_orientation=True)
faces = list(topology.faces())
edges = list(topology.edges())
```

`face_labels[i]` labels `faces[i]`; `edge_labels[j]` labels `edges[j]`.
Indices start at zero and include every primitive in this traversal. Repeated
orientations of the same primitive are deduplicated with `IsSame`.
Read the original STEP as a compound without normalization, healing, or
re-export before matching indices. Indices are traversal positions, not STEP
entity numbers; changing the OCC version or traversal can change them.

The supplied reader validates label lengths and binary values:

```bash
python scripts/read_grounding_labels.py --dataset datasets/grounding --id 00133967_chamfer_6
```

## BRepGround training and evaluation data

Generate the model's graph cache locally:

```bash
python scripts/prepare_grounding.py --dataset datasets/grounding --output datasets/grounding_cache --workers 8
python scripts/validate_grounding_data.py
bash scripts/evaluate_grounding.sh
python scripts/report_metrics.py outputs/grounding-test.json
```

The preprocessing carries labels through shape normalization using OCC
transformation history. The cache is specific to BRepGround; it is not part
of the dataset format. Evaluation reports micro F1, mAP and recall at 3/5/10,
excluding padding.

## Text-to-CAD evaluation data

The evaluation script reads `detail_description` directly from the test split
of `jhlee11/FutureCAD` on Hugging Face and matches rows to the selected CAD IDs.
No manual description export is needed. Downloads use the Hugging Face cache.
Start the BRepGround service before generation. Normalized reference `<id>.ply`
point clouds are also required for geometry evaluation.

```bash
GT_PLY_DIR=/path/to/normalized_plys \
bash scripts/evaluate_llm.sh
python scripts/report_metrics.py outputs/llm-test.json
```

Use `SPLIT=val` for the validation split and `DESCRIPTION_LEVEL=overall` for
overall descriptions. For custom inputs, the optional `DESCRIPTIONS` variable
accepts a local JSON mapping CAD IDs to text strings. Existing `PREDICTIONS`
skip generation and description loading; predictions map IDs to lists of Python
programs. Use a separate predictions file for each split or description level.

The report includes invalid rate (IR), squared bidirectional Chamfer distance
(CD, using 2,048 sampled points), and operation-count G-F1/R-F1 for the Standard
and Advanced subsets. Each generation is evaluated separately. The report keeps
per-example errors and count-F1 inclusion flags; `report_metrics.py` converts
fractions to percentages and scales CD by 1,000.
