from typing import Union, Tuple

import requests
from OCC.Core.TopoDS import TopoDS_Solid, TopoDS_Compound
from OCP.BRepTools import BRepTools
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Builder as OCC_BRep_Builder
import tempfile
import os
from cq_monkey_patch._brep_to_graph import build_graph_as_dict
from occwl.solid import Solid

SERVER_URL = os.environ.get("FUTURECAD_GROUNDING_URL", "http://127.0.0.1:5678").rstrip(
    "/"
)


def ocp_to_pythonocc(ocp_compound):
    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as f:
        temp_path = f.name

    try:
        BRepTools.Write_s(ocp_compound, temp_path)
        occ_compound = TopoDS_Compound()
        builder = OCC_BRep_Builder()
        breptools.Read(occ_compound, temp_path, builder)
    finally:
        os.unlink(temp_path)
    return occ_compound


class Query:
    def __init__(self, query_string):
        self.query_string = query_string

    def query(self, solid: Union[TopoDS_Compound, TopoDS_Solid]) -> Tuple[list, list]:
        solid = ocp_to_pythonocc(solid)
        solid = Solid(solid, allow_compound=True)
        graph_dict, face_bboxes, edge_bboxes = build_graph_as_dict(solid)

        request_data = {
            "graph": graph_dict,
            "query": self.query_string,
        }

        response = requests.post(
            f"{SERVER_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=float(os.environ.get("FUTURECAD_GROUNDING_TIMEOUT", "30")),
        )
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            labels = result["labels"]
            assert len(labels) == (len(face_bboxes) + len(edge_bboxes)), (
                "Number of labels does not match number of faces and edges."
            )
            target_face_bbox = []
            target_edge_bbox = []
            for idx, label in enumerate(labels):
                if label == 1:
                    if idx < len(face_bboxes):
                        target_face_bbox.append(face_bboxes[idx])
                    else:
                        edge_idx = idx - len(face_bboxes)
                        target_edge_bbox.append(edge_bboxes[edge_idx])
            return target_face_bbox, target_edge_bbox

        raise ValueError(f"Prediction failed: {result.get('error')}")
