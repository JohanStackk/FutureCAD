import networkx as nx
from OCC.Core._BRepBndLib import brepbndlib_Add
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.entity_mapper import EntityMapper
from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box


def compute_bbox(shape):
    """
    shape: TopoDS_Shape
    return: {"min": [x,y,z], "max": [x,y,z]}
    """
    box = Bnd_Box()
    brepbndlib_Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return {"min": [xmin, ymin, zmin], "max": [xmax, ymax, zmax]}


def bbox_hash(bbox, digits=4):
    scale = 10**digits

    def q(v):
        return int(round(v * scale))

    return (
        q(bbox["min"][0]),
        q(bbox["min"][1]),
        q(bbox["min"][2]),
        q(bbox["max"][0]),
        q(bbox["max"][1]),
        q(bbox["max"][2]),
    )


def bbox_corners(bbox):
    xmin, ymin, zmin = bbox["min"]
    xmax, ymax, zmax = bbox["max"]
    return [
        (xmin, ymin, zmin),
        (xmin, ymin, zmax),
        (xmin, ymax, zmin),
        (xmin, ymax, zmax),
        (xmax, ymin, zmin),
        (xmax, ymin, zmax),
        (xmax, ymax, zmin),
        (xmax, ymax, zmax),
    ]


def transform_bbox(bbox, trsf):
    pts = []
    for x, y, z in bbox_corners(bbox):
        p = gp_Pnt(x, y, z)
        p.Transform(trsf)
        pts.append((p.X(), p.Y(), p.Z()))
    xs, ys, zs = zip(*pts)
    return {"min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]}


def face_adjacency_infer(shape, trsf_to_original):
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)
    graph = nx.DiGraph()

    face_idx_order = []
    edge_order = []
    face_bbox_list = []
    merged_edge_bbox_list = []

    for face in shape.faces():
        face_idx = mapper.face_index(face)
        face_idx_order.append(face_idx)
        graph.add_node(face_idx, face=face)

        face_bbox_norm = compute_bbox(face.topods_shape())
        face_bbox = transform_bbox(face_bbox_norm, trsf_to_original)
        face_bbox_list.append(face_bbox)

    undirected_key_to_uid = {}
    merged_edge_labels = []
    edge_to_undirected = []
    added_edge_set = set()

    for edge in shape.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(shape.faces_from_edge(edge))
        if len(connected_faces) != 2:
            continue

        left_face, right_face = edge.find_left_and_right_faces(connected_faces)
        if left_face is None or right_face is None:
            continue

        left_index = mapper.face_index(left_face)
        right_index = mapper.face_index(right_face)

        if (left_index, right_index) in added_edge_set or (
            right_index,
            left_index,
        ) in added_edge_set:
            continue

        edge_idx = mapper.oriented_edge_index(edge)
        edge_reversed = edge.reversed_edge()
        if not mapper.oriented_edge_exists(edge_reversed):
            print("[Info] oriented reversed edge does not exist")
            continue
        edge_reversed_idx = mapper.oriented_edge_index(edge_reversed)

        edge_bbox_norm = compute_bbox(edge.topods_shape())
        ehash = bbox_hash(edge_bbox_norm, digits=4)
        selected = 0

        edge_bbox = transform_bbox(edge_bbox_norm, trsf_to_original)
        merged_edge_bbox_list.append(edge_bbox)

        if ehash not in undirected_key_to_uid:
            undirected_key_to_uid[ehash] = len(merged_edge_labels)
            merged_edge_labels.append(selected)
        uid = undirected_key_to_uid[ehash]

        graph.add_edge(
            left_index,
            right_index,
            edge=edge,
            edge_index=edge_idx,
            edge_to_undirected=uid,
        )
        added_edge_set.add((left_index, right_index))
        edge_order.append((left_index, right_index))
        edge_to_undirected.append(uid)

        graph.add_edge(
            right_index,
            left_index,
            edge=edge_reversed,
            edge_index=edge_reversed_idx,
            edge_to_undirected=uid,
        )
        added_edge_set.add((right_index, left_index))
        edge_order.append((right_index, left_index))
        edge_to_undirected.append(uid)

    num_undirected_edges = len(merged_edge_labels)

    labels = {
        "merged_edge_labels": merged_edge_labels,
        "edge_to_undirected": edge_to_undirected,
        "num_undirected_edges": num_undirected_edges,
        "face_bboxes": face_bbox_list,
        "merged_edge_bboxes": merged_edge_bbox_list,
    }

    used = set(edge_to_undirected)
    assert len(used) == len(merged_edge_labels), (
        f"Found orphan undirected ids: used={len(used)} total={len(merged_edge_labels)}"
    )
    assert len(face_bbox_list) == graph.number_of_nodes(), (
        f"#face-bboxes: {len(face_bbox_list)} does not match #graph-nodes: {graph.number_of_nodes()}"
    )
    assert len(merged_edge_bbox_list) == graph.number_of_edges() // 2, (
        f"#edge-bboxes: {len(merged_edge_bbox_list)} does not match #graph-edges/2: {graph.number_of_edges() // 2}"
    )
    assert max(used) == len(merged_edge_labels) - 1
    assert min(used) == 0

    return graph, labels, face_idx_order, edge_order
