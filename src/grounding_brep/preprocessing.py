import dgl
import numpy as np
import torch
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import ugrid, uvgrid

from grounding_brep.edge_geometry import edge_convexity


def normalize_solid(solid: Solid, raw_labels):
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    shape = solid.topods_shape()
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    (x_min, y_min, z_min, x_max, y_max, z_max) = bbox.Get()
    min_corner = np.array([x_min, y_min, z_min])
    max_corner = np.array([x_max, y_max, z_max])
    extents = max_corner - min_corner
    max_range = float(extents.max())
    max_range = max(max_range, 1e-12)
    scale = 2.0 / max_range
    tr_scale = gp_Trsf()
    tr_scale.SetScale(gp_Pnt(0, 0, 0), scale)
    scale_builder = BRepBuilderAPI_Transform(shape, tr_scale, True)
    scaled_shape = scale_builder.Shape()
    bbox2 = Bnd_Box()
    brepbndlib.Add(scaled_shape, bbox2)
    (xs0, ys0, zs0, xs1, ys1, zs1) = bbox2.Get()
    center_scaled = [(xs0 + xs1) * 0.5, (ys0 + ys1) * 0.5, (zs0 + zs1) * 0.5]
    tr_move = gp_Trsf()
    tr_move.SetTranslation(
        gp_Vec(-center_scaled[0], -center_scaled[1], -center_scaled[2])
    )
    move_builder = BRepBuilderAPI_Transform(scaled_shape, tr_move, True)
    normalized_shape = move_builder.Shape()
    trsf_norm = tr_move
    trsf_norm.Multiply(tr_scale)
    normalized = Solid(normalized_shape, allow_compound=True)
    mapped = {"query": raw_labels["query"]}
    for kind, field in (("faces", "face_labels"), ("edges", "edge_labels")):
        lookup = TopTools_IndexedMapOfShape()
        primitives = list(getattr(normalized, kind)())
        for primitive in primitives:
            lookup.Add(primitive.topods_shape())
        values = [None] * len(primitives)
        originals = list(getattr(solid, kind)())
        if len(originals) != len(raw_labels[field]):
            raise ValueError(f"Original topology disagrees with {field}")
        for primitive, flag in zip(originals, raw_labels[field]):
            transformed = move_builder.ModifiedShape(
                scale_builder.ModifiedShape(primitive.topods_shape())
            )
            index = lookup.FindIndex(transformed) - 1
            if index < 0 or values[index] is not None:
                raise ValueError(f"Normalization changed {kind} identity")
            values[index] = flag
        if any(value is None for value in values):
            raise ValueError(f"Incomplete normalized {kind} mapping")
        mapped[field] = values
    return normalized, trsf_norm, mapped


def get_face_area(face: Face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face.topods_shape(), props)
    return float(props.Mass())


def get_face_type(face: Face):
    return face.surface_type_enum()


def get_face_centroid(face: Face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face.topods_shape(), props)
    com = props.CentreOfMass()
    return np.array([com.X(), com.Y(), com.Z()], dtype=np.float32)


def get_face_rational_flag(face):
    surface = BRepAdaptor_Surface(face.topods_shape())
    return float(surface.IsURational() or surface.IsVRational())


def get_edge_length(edge: Edge):
    return float(edge.length())


def get_edge_type(edge: Edge):
    return edge.curve_type_enum()


def compute_edge_convexity(edge: Edge, face1: Face, face2: Face, solid: Solid):
    convexity = edge_convexity(
        edge.topods_shape(),
        face1.topods_shape(),
        face2.topods_shape(),
        solid.topods_shape(),
    )
    if convexity == "convex":
        return np.array([1, 0, 0], dtype=np.float32)
    elif convexity == "concave":
        return np.array([0, 1, 0], dtype=np.float32)
    else:
        return np.array([0, 0, 1], dtype=np.float32)


def build_graph(
    solid,
    raw_labels,
    curv_u=32,
    surf_u=32,
    surf_v=32,
    num_face_types=11,
    num_edge_types=9,
):
    solid, trsf_norm, raw_labels = normalize_solid(solid, raw_labels)
    trsf_to_original = trsf_norm.Inverted()
    (graph, labels, face_idx_order, edge_order) = training_adjacency(
        solid, raw_labels, trsf_to_original
    )
    face_feats = []
    face_area = []
    face_type_oh = []
    face_centroid = []
    face_rational = []
    for face_idx in face_idx_order:
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(face, method="point", num_u=surf_u, num_v=surf_v)
        normals = uvgrid(face, method="normal", num_u=surf_u, num_v=surf_v)
        vis = uvgrid(face, method="visibility_status", num_u=surf_u, num_v=surf_v)
        mask = np.logical_or(vis == 0, vis == 2)
        cnn_feat = np.concatenate((points, normals, mask), axis=-1)
        face_feats.append(cnn_feat)
        area = get_face_area(face)
        ftype = get_face_type(face)
        centroid = get_face_centroid(face)
        rational = get_face_rational_flag(face)
        ftype_oh = np.zeros(num_face_types, dtype=np.float32)
        ftype_oh[ftype] = 1.0
        face_area.append([area])
        face_type_oh.append(ftype_oh)
        face_centroid.append(centroid)
        face_rational.append([rational])
    face_feats = np.asarray(face_feats, dtype=np.float32)
    face_area = np.asarray(face_area, dtype=np.float32)
    face_type_oh = np.asarray(face_type_oh, dtype=np.float32)
    face_centroid = np.asarray(face_centroid, dtype=np.float32)
    face_rational = np.asarray(face_rational, dtype=np.float32)
    edge_feats = []
    edge_length = []
    edge_type_oh = []
    edge_convexity = []
    for u, v in edge_order:
        edge = graph.edges[u, v]["edge"]
        if not edge.has_curve():
            raise RuntimeError("No curve")
        pts = ugrid(edge, method="point", num_u=curv_u)
        tans = ugrid(edge, method="tangent", num_u=curv_u)
        cnn_feat = np.concatenate((pts, tans), axis=-1)
        edge_feats.append(cnn_feat)
        length = get_edge_length(edge)
        etype = get_edge_type(edge)
        etype_oh = np.zeros(num_edge_types, dtype=np.float32)
        etype_oh[etype] = 1.0
        face1 = graph.nodes[u]["face"]
        face2 = graph.nodes[v]["face"]
        conv = compute_edge_convexity(edge, face1, face2, solid)
        edge_length.append([length])
        edge_type_oh.append(etype_oh)
        edge_convexity.append(conv)
    edge_feats = np.asarray(edge_feats, dtype=np.float32)
    edge_length = np.asarray(edge_length, dtype=np.float32)
    edge_type_oh = np.asarray(edge_type_oh, dtype=np.float32)
    edge_convexity = np.asarray(edge_convexity, dtype=np.float32)
    src = [u for (u, v) in edge_order]
    dst = [v for (u, v) in edge_order]
    g = dgl.graph((src, dst), num_nodes=len(face_idx_order))
    g.ndata["x"] = torch.from_numpy(face_feats)
    g.edata["x"] = torch.from_numpy(edge_feats)
    g.ndata["area"] = torch.from_numpy(face_area)
    g.ndata["ftype"] = torch.from_numpy(face_type_oh)
    g.ndata["centroid"] = torch.from_numpy(face_centroid)
    g.ndata["rational"] = torch.from_numpy(face_rational)
    g.edata["length"] = torch.from_numpy(edge_length)
    g.edata["etype"] = torch.from_numpy(edge_type_oh)
    g.edata["convexity"] = torch.from_numpy(edge_convexity)
    edge_to_undirected = np.asarray(labels["edge_to_undirected"], dtype=np.int64)
    g.edata["edge_to_undirected"] = torch.from_numpy(edge_to_undirected)
    assert g.num_edges() == len(edge_order)
    assert g.edata["edge_to_undirected"].numel() == g.num_edges()
    assert g.num_nodes() == len(face_idx_order), (
        f"#nodes {g.num_nodes()} != #face_idx_order {len(face_idx_order)}"
    )
    assert g.ndata["x"].shape[0] == g.num_nodes(), (
        f"face CNN feats {g.ndata['x'].shape[0]} != #nodes {g.num_nodes()}"
    )
    uid = g.edata["edge_to_undirected"]
    assert torch.all(uid[0::2] == uid[1::2]), (
        "adjacent edges are not paired undirected edges"
    )
    return (g, labels)


def training_adjacency(solid, raw_labels, to_original):
    import networkx as nx
    from occwl.entity_mapper import EntityMapper
    from occwl.graph import bbox_hash, compute_bbox, transform_bbox

    mapper = EntityMapper(solid)
    faces, edges = list(solid.faces()), list(solid.edges())
    if len(faces) != len(raw_labels["face_labels"]) or len(edges) != len(
        raw_labels["edge_labels"]
    ):
        raise ValueError("STEP topology disagrees with primitive labels")
    graph = nx.DiGraph()
    for i, face in enumerate(faces):
        if mapper.face_index(face) != i:
            raise ValueError("Face traversal order changed")
        graph.add_node(i, face=face)
    groups, selected, mapping, order, seen = {}, [], [], [], set()
    for i, edge in enumerate(edges):
        if not edge.has_curve():
            continue
        connected = list(solid.faces_from_edge(edge))
        if len(connected) != 2:
            continue
        left, right = edge.find_left_and_right_faces(connected)
        if left is None or right is None:
            continue
        u, v = mapper.face_index(left), mapper.face_index(right)
        if (u, v) in seen or (v, u) in seen:
            continue
        reverse = edge.reversed_edge()
        if not mapper.oriented_edge_exists(reverse):
            continue
        key = bbox_hash(
            transform_bbox(compute_bbox(edge.topods_shape()), to_original), digits=4
        )
        if key not in groups:
            groups[key] = len(selected)
            selected.append(0)
        uid = groups[key]
        selected[uid] |= raw_labels["edge_labels"][i]
        graph.add_edge(u, v, edge=edge)
        graph.add_edge(v, u, edge=reverse)
        order.extend([(u, v), (v, u)])
        mapping.extend([uid, uid])
        seen.update([(u, v), (v, u)])
    if len(order) != 2 * len(selected):
        raise ValueError(
            "This topology requires a multigraph; the BRepGround model expects paired edges"
        )
    labels = {
        "query": raw_labels["query"],
        "face_labels": raw_labels["face_labels"],
        "merged_edge_labels": selected,
        "edge_to_undirected": mapping,
        "num_undirected_edges": len(selected),
        "labels": raw_labels["face_labels"] + selected,
    }
    return graph, labels, list(range(len(faces))), order
