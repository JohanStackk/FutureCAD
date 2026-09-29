# Render a CadQuery program

The input is a Python program defining the CadQuery object `model`.
Execute it and export STEP, STL, and an isometric SVG view:

To render a FutureCAD modeling-history JSON, first convert it to a point-selection
program:

```bash
python -m futurecad.convert --input model.json --selection points --output-dir outputs/converted
python visualize/render.py --program outputs/converted/points/model.py --output-dir outputs/model-view
```

To render an existing generated program:

```bash
conda activate futurecad
python visualize/render.py --program outputs/plate-demo/cad/demo/sample0/model.py --output-dir outputs/plate-view
```

Open `outputs/plate-view/model.svg` in a browser to view the result.
SVG rendering uses CadQuery's exporter and does not need a GUI or display server.
Point-selection programs execute without a BRepGround service. Programs using
`cq.Query` with `filletQ`, `chamferQ`, or `shellQ` need `cq_monkey_patch` and the
BRepGround service; start `bash scripts/serve_grounding.sh` first. See
[training target formats](../docs/dataset.md#text-to-cad-training-targets) for
examples of both forms and the dataset conversion commands.
