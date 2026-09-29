import json
from pathlib import Path


def read_step_labels(step_path, labels_path):
    from OCC.Extend.DataExchange import read_step_file
    from OCC.Extend.TopologyUtils import TopologyExplorer

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    shape = read_step_file(str(step_path), as_compound=True, verbosity=False)
    topology = TopologyExplorer(shape, ignore_orientation=True)
    faces = list(topology.faces())
    edges = list(topology.edges())
    for kind, primitives in (("face_labels", faces), ("edge_labels", edges)):
        values = labels[kind]
        if len(values) != len(primitives):
            raise ValueError(f"{kind} count disagrees with STEP: {step_path}")
        if any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError(f"Invalid binary values in {kind}")
    if not labels["query"] or any(not isinstance(q, str) or not q.strip() for q in labels["query"]):
        raise ValueError("Expected nonempty text queries")
    return {"shape": shape, "faces": faces, "edges": edges, "labels": labels}
