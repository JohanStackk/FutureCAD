# FutureCAD

**Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding** · ICML 2026 Spotlight

[Paper](https://arxiv.org/abs/2603.11831) · [FutureCAD dataset](https://huggingface.co/datasets/jhlee11/FutureCAD) · [BRepGround dataset](https://huggingface.co/datasets/jhlee11/BRepGround)

[FutureCAD-7B model](https://huggingface.co/jhlee11/FutureCAD-7B) · [BRepGround model](https://huggingface.co/jhlee11/BRepGround)

Code for text-to-CAD generation, BRepGround training and inference, evaluation, and the veRL reward function.

## Setup

Follow [Environment setup](docs/environment.md) to create the `futurecad` environment. Run the commands below from the repository root.

```bash
python scripts/download_models.py
```

Use `--model grounding` or `--model llm` to download one model.

The entry points use these paths:

```text
weights/brepground/model.safetensors
weights/brepground/bert-large-uncased/
weights/futurecad-llm/
```

## Text-to-CAD

Start BRepGround in one terminal:

```bash
conda activate futurecad
bash scripts/serve_grounding.sh
```

In another terminal:

```bash
conda activate futurecad
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m futurecad.text_to_cad \
  --text-file examples/prompt.txt \
  --model weights/futurecad-llm \
  --output-dir outputs/plate-demo
```

Alternatively, use `--text "Create a cylinder with radius 10 mm and height 20 mm."`.
The output includes `predictions.json`, the generated `cad/demo/sample0/model.py`,
and `model.step` / `model.stl` when CAD execution succeeds. Errors are recorded in
`run.json` and `cad/exports.json`. Use a new output directory for each run.

To execute an existing program without loading the LLM:

```bash
python -m futurecad.export --predictions examples/programs.json --output-dir outputs/example-cad
```

The grounding endpoint defaults to `http://127.0.0.1:5678`; set
`FUTURECAD_GROUNDING_URL` to use another endpoint.

To render a CadQuery program as an SVG view, use
[visualize/render.py](visualize/README.md).

## Text-to-CAD training data

[FutureCAD](https://huggingface.co/datasets/jhlee11/FutureCAD) provides modeling
histories and `overall_description` / `detail_description`. An LLM training
pair consists of one description as input and the complete CadQuery program
converted from the corresponding history JSON as the target.

For fillet, chamfer, and shell, point-selection programs support local geometry
validation; the paper's BRep-primitive-query programs use BRepGround with
`cq_monkey_patch`. See [data download and training target formats](docs/dataset.md#text-to-cad-training-targets)
for both forms and the available files.

Convert a dataset sample and render the point-selection program:

```bash
python -m futurecad.convert --split test --id 00114630 --selection both --output-dir outputs/converted-example
python visualize/render.py --program outputs/converted-example/points/00114630.py --output-dir outputs/converted-view
```

Generate the query-format programs and training pairs directly from Hugging Face:

```bash
python -m futurecad.convert --split train --selection query --training-pairs --output-dir outputs/training-query
```

## BRepGround

Download STEP files and query labels from Hugging Face:

```bash
hf download jhlee11/BRepGround export_data.py --repo-type dataset --local-dir downloads/brepground
python downloads/brepground/export_data.py --split all --output datasets/grounding
```

Labels correspond to faces and edges read with **pythonocc-core 7.8.1.1 / OCCT 7.8.1**.
See [label indexing](docs/dataset.md) for the traversal convention and reader.
Build the network's graph cache locally:

```bash
conda activate futurecad
python scripts/prepare_grounding.py --workers 8
python scripts/validate_grounding_data.py
```

Run inference and calculate metrics on the test split:

```bash
bash scripts/evaluate_grounding.sh
python scripts/report_metrics.py outputs/grounding-test.json
```

Evaluation and serving use `configs/brepground-inference.yaml`.

Train from scratch, initializing the text encoder from BERT:

```bash
hf download google-bert/bert-large-uncased --local-dir weights/brepground/bert-large-uncased
python -m grounding_brep.train --config configs/brepground.yaml
```

Continue training from a checkpoint:

```bash
python -m grounding_brep.train --config configs/brepground.yaml \
  --initialize-from weights/brepground/model.safetensors
```

Use `--resume` with a Lightning training checkpoint to restore optimizer state.
Validation selects training checkpoints. Configure data paths, batch sizes and
training settings in `configs/brepground.yaml` or pass dot-list overrides such as
`data.batch_size=8`. Split IDs are in `data/splits/grounding.json`.

## LLM generation and evaluation

The evaluation script reads `detail_description` directly from the FutureCAD
test split on Hugging Face. With the grounding service running:

```bash
conda activate futurecad
export VLLM_USE_FLASHINFER_SAMPLER=0
GT_PLY_DIR=/path/to/normalized_plys \
bash scripts/evaluate_llm.sh
python scripts/report_metrics.py outputs/llm-test.json
```

Set `SPLIT=val` for validation or `DESCRIPTION_LEVEL=overall` for overall
descriptions. `DESCRIPTIONS=/path/to/descriptions.json` optionally overrides the
dataset descriptions with a local ID-to-text JSON.

The reference directory contains
normalized `<id>.ply` point clouds, provided separately from the published
modeling histories and descriptions. The repository includes
train/val/test IDs, Advanced subset IDs and operation-count references.

To evaluate existing predictions, set `PREDICTIONS=/path/to/predictions.json`.
Predictions map each ID to a list of Python programs; their IDs must match the
selected split. Reports contain CD, G-F1, R-F1 and IR for the Standard and Advanced
subsets, with each generation evaluated separately.
See [dataset usage](docs/dataset.md) for input formats and evaluation commands.

## veRL reward

We use veRL with GSPO for reinforcement learning and provide the
[reward function definition](src/futurecad/rewards/cad_reward.py).
For veRL setup and training instructions, see the
[official veRL repository](https://github.com/verl-project/verl).

## Layout

```text
configs/                  model, training and inference settings
data/                     splits and evaluation references
environments/             installation recipes
examples/                 text prompt and executable CAD program
src/grounding_brep/        preprocessing, model, training, inference and serving
src/futurecad/             generation, execution, reward and metrics
src/cq_monkey_patch/       CadQuery text-query and refinement extensions
src/occwl/                 adapted OCC wrapper
scripts/                  data preparation and command-line launchers
visualize/                CadQuery solid export and SVG rendering
tests/                    regression tests
docs/                     environment and dataset usage
```

## Tests

```bash
python scripts/validate_data.py
python -m unittest discover -s tests -v
```

The CAD tests require CadQuery. GPU inference and training require the `futurecad`
environment and local model files. CAD programs execute in subprocesses with a
timeout; these subprocesses are not a security sandbox.

## occwl and acknowledgments

We use a locally adapted copy of [Autodesk's occwl](https://github.com/AutodeskAILab/occwl).
The OpenCascade/pythonocc version used by upstream occwl could not be aligned with
CadQuery's OCP version in our development environment, so we cloned the library and
adapted its APIs. The bridge transfers shapes between the bindings through BRep
serialization. Use the bundled copy in `src/occwl`.

We thank **Pradeep Kumar Jayaraman, Joseph Lambourne, and the Autodesk AI Lab
contributors** for occwl. The encoder builds on
[UV-Net](https://github.com/AutodeskAILab/UV-Net), with its
[MIT license](LICENSES/UV-Net.txt) included. See [third-party notices](docs/third_party.md).

The authors' code is released under the [MIT license](LICENSE).

## Citation

```bibtex
@misc{li2026highfidelitycadgenerationllmdriven,
  title={Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding},
  author={Jiahao Li and Qingwang Zhang and Qiuyu Chen and Guozhan Qiu and Yunzhong Lou and Xiangdong Zhou},
  year={2026},
  eprint={2603.11831},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.11831},
}
```
