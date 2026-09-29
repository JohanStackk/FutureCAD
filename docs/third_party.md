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
