# Third-party notices

## UV-Net

`src/grounding_brep/models/uv_net` derives from
[AutodeskAILab/UV-Net](https://github.com/AutodeskAILab/UV-Net).
Its MIT license is preserved in [LICENSES/UV-Net.txt](../LICENSES/UV-Net.txt).

## occwl

`src/occwl` is an adapted copy of
[AutodeskAILab/occwl](https://github.com/AutodeskAILab/occwl).
We thank Pradeep Kumar Jayaraman, Joseph Lambourne, and the Autodesk AI Lab
contributors. Differences between the upstream OpenCascade/pythonocc version
and CadQuery's OCP version required local API changes. The CAD bridge transfers
shapes through BRep serialization.

The inspected upstream repository does not provide an explicit license file.
The FutureCAD MIT license does not grant rights to this third-party code.

## FutureCAD

The authors' code is available under the [MIT license](../LICENSE).
The [FutureCAD](https://huggingface.co/datasets/jhlee11/FutureCAD) and
[BRepGround](https://huggingface.co/datasets/jhlee11/BRepGround) datasets use
CC BY 4.0. Third-party dependencies retain their own licenses.

## veRL reward manager

`integrations/verl/naive.py` adapts the
[veRL 0.6.1 reward manager](https://github.com/verl-project/verl/blob/v0.6.1/verl/workers/reward_manager/naive.py)
from the veRL contributors and ByteDance. It replaces serial CAD scoring with
bounded process-based concurrency. The file preserves the upstream copyright
and Apache 2.0 notice; the full license is in
[LICENSES/veRL.txt](../LICENSES/veRL.txt).
