# Environment setup

Use one `futurecad` environment for CAD execution, BRepGround preprocessing,
training and serving, and LLM inference. The GPU setup targets Linux x86-64
with an NVIDIA GPU and glibc 2.31 or newer (for example, Ubuntu 20.04+).

Run from the repository root:

```bash
conda env create -f environments/futurecad.yml
conda activate futurecad
python -m pip install -r environments/requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/check_environment.py --component all
```

The setup uses PyTorch 2.7.1, vLLM 0.10.0 and DGL 1.1.3. DGL 1.1.3 provides
the graph operations used by BRepGround without the GraphBolt dependency.
Use the supplied [CUDA 12.1 DGL wheel](https://data.dgl.ai/wheels/cu121/repo.html);
vLLM's [CUDA dependencies](https://github.com/vllm-project/vllm/blob/v0.10.0/requirements/cuda.txt)
pin PyTorch 2.7.1. CUDA libraries are installed by the wheels; the host needs
a compatible NVIDIA driver.

The check exercises the CadQuery/pythonocc bridge, CUDA graph operations and
the vLLM import. The LLM and grounding service run in separate processes within
this same environment. No alternate Python paths are needed in the examples.
Set `CUDA_VISIBLE_DEVICES` to choose the GPU and
`VLLM_USE_FLASHINFER_SAMPLER=0` to use PyTorch sampling.

The STEP reader uses pythonocc-core 7.8.1.1 / OCCT 7.8.1. Their Conda builds
exclude VTK so CadQuery can install its matching VTK wheel. Use the bundled
`src/occwl`, whose APIs are adapted for this project.
