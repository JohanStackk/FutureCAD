import numpy as np
import torch
from grounding_brep.edge_geometry import edge_convexity
from OCC.Core._BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from occwl.edge import Edge
from occwl.face import Face
from occwl.graph import face_adjacency_infer
from occwl.solid import Solid
from occwl.uvgrid import ugrid, uvgrid

FILTER_CONDITIONS = {
    "edata.length": 6.3,
    "edata.x": 1.01,
    "ndata.area": 8.5,
    "ndata.centroid": 1.01,
    "ndata.x": 1.01,
}

DEFAULT_FILL_VALUES = {
    "edata.length": 0.0,
    "edata.x": 0.0,
    "ndata.area": 0.0,
    "ndata.centroid": 0.0,
    "ndata.x": 0.0,
}


def sanitize_tensor(t: torch.Tensor, key: str, verbose: bool = False) -> torch.Tensor:
    t = t.float()

    threshold = FILTER_CONDITIONS.get(key, None)
    fill_value = DEFAULT_FILL_VALUES.get(key, 0.0)

    nan_mask = torch.isnan(t)
    nan_count = nan_mask.sum().item()
    if nan_count > 0:
        t = t.masked_fill(nan_mask, fill_value)
        if verbose:
            print(f"  [{key}] Fixed {nan_count} NaN values -> {fill_value}")

    inf_mask = torch.isinf(t)
    inf_count = inf_mask.sum().item()
    if inf_count > 0:
        t = t.masked_fill(inf_mask, fill_value)
        if verbose:
            print(f"  [{key}] Fixed {inf_count} Inf values -> {fill_value}")

    if threshold is not None:
        outlier_mask = t.abs() > threshold
        outlier_count = outlier_mask.sum().item()
        if outlier_count > 0:
            t = t.clamp(-threshold, threshold)
            if verbose:
                print(
                    f"  [{key}] Clipped {outlier_count} outliers to [-{threshold}, {threshold}]"
                )
    return t


def sanitize_array(arr: np.ndarray, key: str, verbose: bool = False) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32))
    t = sanitize_tensor(t, key, verbose)
    return t.numpy()


def sanitize_scalar(value: float, key: str, verbose: bool = False) -> float:
    threshold = FILTER_CONDITIONS.get(key, None)
    fill_value = DEFAULT_FILL_VALUES.get(key, 0.0)

    import math

    if math.isnan(value) or math.isinf(value):
        if verbose:
            print(f"  [{key}] Fixed NaN/Inf scalar -> {fill_value}")
        return fill_value

    if threshold is not None and abs(value) > threshold:
        clipped = max(-threshold, min(threshold, value))
        if verbose:
            print(f"  [{key}] Clipped scalar {value} -> {clipped}")
        return clipped
    return value


def normalize_solid(solid: Solid):
    shape = solid.topods_shape()
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
    min_corner = np.array([x_min, y_min, z_min])
    max_corner = np.array([x_max, y_max, z_max])

    extents = max_corner - min_corner
    max_range = float(extents.max())
    max_range = max(max_range, 1e-12)
    scale = 2.0 / max_range

    tr_scale = gp_Trsf()
    tr_scale.SetScale(gp_Pnt(0, 0, 0), scale)
    scaled_shape = BRepBuilderAPI_Transform(shape, tr_scale, True).Shape()

    bbox2 = Bnd_Box()
    brepbndlib.Add(scaled_shape, bbox2)
    xs0, ys0, zs0, xs1, ys1, zs1 = bbox2.Get()
    center_scaled = [(xs0 + xs1) * 0.5, (ys0 + ys1) * 0.5, (zs0 + zs1) * 0.5]

    tr_move = gp_Trsf()
    tr_move.SetTranslation(
        gp_Vec(-center_scaled[0], -center_scaled[1], -center_scaled[2])
    )
    normalized_shape = BRepBuilderAPI_Transform(scaled_shape, tr_move, True).Shape()

    trsf_norm = tr_move
    trsf_norm.Multiply(tr_scale)

    return Solid(normalized_shape, allow_compound=True), trsf_norm


def get_face_area(face: Face):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face.topods_shape(), props)
    return float(props.Mass())


def get_face_type(face: Face):
    return face.surface_type_enum()


def get_face_centroid(face: Face):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face.topods_shape(), props)
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


def build_graph_as_dict(
    solid,
    curv_u=32,
    surf_u=32,
    surf_v=32,
    num_face_types=11,
    num_edge_types=9,
    verbose=True,
):
    solid, trsf_norm = normalize_solid(solid)
    trsf_to_original = trsf_norm.Inverted()
    graph, labels, face_idx_order, edge_order = face_adjacency_infer(
        solid, trsf_to_original
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
        cnn_feat = sanitize_array(cnn_feat, "ndata.x", verbose)
        face_feats.append(cnn_feat.tolist())

        area = get_face_area(face)
        area = sanitize_scalar(area, "ndata.area", verbose)

        ftype = get_face_type(face)
        centroid = get_face_centroid(face)
        centroid = sanitize_array(centroid, "ndata.centroid", verbose)

        rational = get_face_rational_flag(face)

        ftype_oh = np.zeros(num_face_types, dtype=np.float32)
        ftype_oh[ftype] = 1.0

        face_area.append([area])
        face_type_oh.append(ftype_oh.tolist())
        face_centroid.append(centroid.tolist())
        face_rational.append([rational])

    edge_feats = []
    edge_length = []
    edge_type_oh = []
    edge_convexity_list = []

    for u, v in edge_order:
        edge = graph.edges[(u, v)]["edge"]

        if not edge.has_curve():
            raise RuntimeError("No curve")

        pts = ugrid(edge, method="point", num_u=curv_u)
        tans = ugrid(edge, method="tangent", num_u=curv_u)
        cnn_feat = np.concatenate((pts, tans), axis=-1)
        cnn_feat = sanitize_array(cnn_feat, "edata.x", verbose)
        edge_feats.append(cnn_feat.tolist())

        length = get_edge_length(edge)
        length = sanitize_scalar(length, "edata.length", verbose)

        etype = get_edge_type(edge)

        etype_oh = np.zeros(num_edge_types, dtype=np.float32)
        etype_oh[etype] = 1.0

        face1 = graph.nodes[u]["face"]
        face2 = graph.nodes[v]["face"]
        conv = compute_edge_convexity(edge, face1, face2, solid)

        edge_length.append([length])
        edge_type_oh.append(etype_oh.tolist())
        edge_convexity_list.append(conv.tolist())

    edge_to_undirected = labels["edge_to_undirected"]

    src = [u for (u, v) in edge_order]
    dst = [v for (u, v) in edge_order]

    graph_dict = {
        "num_nodes": len(face_idx_order),
        "edges": {"src": src, "dst": dst},
        "ndata": {
            "x": face_feats,
            "area": face_area,
            "ftype": face_type_oh,
            "centroid": face_centroid,
            "rational": face_rational,
        },
        "edata": {
            "x": edge_feats,
            "length": edge_length,
            "etype": edge_type_oh,
            "convexity": edge_convexity_list,
            "edge_to_undirected": edge_to_undirected,
        },
    }
    return graph_dict, labels["face_bboxes"], labels["merged_edge_bboxes"]
