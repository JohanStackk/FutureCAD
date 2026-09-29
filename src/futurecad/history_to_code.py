import ast
import copy

import numpy as np


def convert_history(history, selection="points", query_index=None):
    if selection not in {"points", "query"}:
        raise ValueError("Selection must be points or query")
    if query_index is not None and (
        selection != "query" or not isinstance(query_index, int) or query_index < 0
    ):
        raise ValueError(
            "query_index requires query selection and a nonnegative integer"
        )
    history = copy.deepcopy(history)
    features, sequence = history["features"], history["sequence"]
    refinements = []
    has_solid = False
    for item in sequence:
        kind = item["type"]
        if kind not in {"sketch", "extrude", "revolve", "fillet", "chamfer", "shell"}:
            raise ValueError(f"Unsupported feature type: {kind}")
        feature = features[item["feature_id"]]
        if feature["type"] != kind:
            raise ValueError(
                f"Feature type disagrees with sequence: {item['feature_id']}"
            )
        if kind in {"extrude", "revolve"}:
            has_solid = True
        elif kind in {"fillet", "chamfer", "shell"}:
            if not has_solid:
                raise ValueError("A refinement must follow a solid operation")
            refinements.append(feature)
    if not has_solid:
        raise ValueError("History contains no solid operation")
    indices = [None]
    if selection == "query" and refinements:
        queries = [feature.get("query") for feature in refinements]
        if any(
            not isinstance(items, list)
            or not items
            or any(not isinstance(text, str) or not text.strip() for text in items)
            for items in queries
        ):
            raise ValueError("Every refinement must have nonempty text queries")
        count = min(3, min(len(items) for items in queries))
        if query_index is not None and query_index >= count:
            raise ValueError(f"query_index must be less than {count}")
        indices = list(range(count)) if query_index is None else [query_index]
    elif query_index not in {None, 0}:
        raise ValueError("A history without refinements has only one program")
    programs = []
    for index in indices:
        code = "import cq_monkey_patch\nimport cadquery as cq\n\n"
        code += _generate_code_with_refine(
            history, com=selection == "points", query_idx=index
        )
        ast.parse(code)
        programs.append(code)
    return programs


def _line2cq(line_data):
    (e_x, e_y) = line_data["end_point"]
    return f"lineTo({e_x}, {e_y})"


def _arc2cq(arc_data):
    (m_x, m_y) = arc_data["midpoint"]
    (e_x, e_y) = arc_data["end_point"]
    return f"threePointArc(({m_x}, {m_y}), ({e_x}, {e_y}))"


def _circle2cq(circle_data):
    (x, y) = circle_data["center_point"]
    r = circle_data["radius"]
    return f"moveTo({x}, {y}).circle({r})"


def _bspline2cq(bspline_data):
    pts = bspline_data["interpolated_points"][1:]
    pts = [tuple(pt) for pt in pts]
    return f"spline({pts}, includeCurrent=True)"


def _curve2cq(curve):
    t = curve["type"]
    if t in ["Line3D", "Line2D"]:
        return _line2cq(curve)
    if t in ["Arc3D", "Arc2D"]:
        return _arc2cq(curve)
    if t in ["Circle3D", "Circle2D"]:
        return _circle2cq(curve)
    if t in ["BSplineCurve3D", "BSplineCurve2D"]:
        return _bspline2cq(curve)
    raise NotImplementedError(t)


def _vectors_equal(a, b, eps=0.0001):
    return np.linalg.norm(np.array(a) - np.array(b)) < eps


def _workplane_cq(plane, wp_name):
    origin = plane["origin"]
    xDir = plane["x"]
    normal = plane["normal"]
    namedPlanes = {
        "XY": ((1, 0, 0), (0, 0, 1)),
        "-XY": ((1, 0, 0), (0, 0, -1)),
        "YZ": ((0, 1, 0), (1, 0, 0)),
        "-YZ": ((0, 1, 0), (-1, 0, 0)),
        "ZX": ((0, 0, 1), (0, 1, 0)),
        "-ZX": ((0, 0, 1), (0, -1, 0)),
        "XZ": ((1, 0, 0), (0, -1, 0)),
        "-XZ": ((1, 0, 0), (0, 1, 0)),
        "YX": ((0, 1, 0), (0, 0, -1)),
        "-YX": ((0, 1, 0), (0, 0, 1)),
        "ZY": ((0, 0, 1), (-1, 0, 0)),
        "-ZY": ((0, 0, 1), (1, 0, 0)),
        "front": ((1, 0, 0), (0, 0, 1)),
        "back": ((-1, 0, 0), (0, 0, -1)),
        "left": ((0, 0, 1), (-1, 0, 0)),
        "right": ((0, 0, -1), (1, 0, 0)),
        "top": ((1, 0, 0), (0, 1, 0)),
        "bottom": ((1, 0, 0), (0, -1, 0)),
    }
    for name, (x_std, n_std) in namedPlanes.items():
        if (
            _vectors_equal(xDir, x_std)
            and _vectors_equal(normal, n_std)
            and _vectors_equal(origin, (0, 0, 0))
        ):
            return f"{wp_name} = cq.Workplane('{name}')"
    for name, (x_std, n_std) in namedPlanes.items():
        if _vectors_equal(xDir, x_std) and _vectors_equal(normal, n_std):
            return f"{wp_name} = cq.Workplane('{name}', origin={tuple(origin)})"
    return f"{wp_name} = cq.Workplane(cq.Plane(cq.Vector{tuple(origin)}, cq.Vector{tuple(xDir)}, cq.Vector{tuple(normal)}))"


def _loop2cq(curves, wp_name):
    if len(curves) == 1:
        assert curves[0]["type"] in ["Circle3D", "Circle2D"], (
            "Single curve in loop must be a Circle2/3D"
        )
        return f"{wp_name}.{_curve2cq(curves[0])}"
    else:
        first_curve = curves[0]
        (s_x, s_y) = first_curve["start_point"]
        cq_commands = f"{wp_name}.moveTo({s_x}, {s_y})"
        for idx, curve in enumerate(curves):
            if idx == 0:
                continue
            if curve["start_point_id"] != curves[idx - 1]["end_point_id"]:
                tmp_start = curve["start_point"]
                tmp_start_id = curve["start_point_id"]
                curve["start_point"] = curve["end_point"]
                curve["start_point_id"] = curve["end_point_id"]
                curve["end_point"] = tmp_start
                curve["end_point_id"] = tmp_start_id
        for curve in curves:
            assert curve["type"] not in ["Circle3D", "Circle2D"], (
                "Circle2/3D cannot be part of a multi-curve loop"
            )
            cq_commands += f".{_curve2cq(curve)}"
        cq_commands += ".close()"
        return cq_commands


def _extrude2cq(extrude_data, wp_name, shape_name):
    depth_one = extrude_data["parameters"]["depthOne"]
    depth_two = extrude_data["parameters"]["depthTwo"]
    assert depth_one != 0 or depth_two != 0, (
        "At least one of depthOne or depthTwo must be non-zero"
    )
    cq_commands = (
        f"{shape_name} = {wp_name}.extrude_bidir(distance=({depth_one}, {depth_two}))"
    )
    return cq_commands


def _revolve2cq(revolve_data, wp_name, shape_name):
    angle_one = revolve_data["parameters"]["angleOne"]
    angle_two = revolve_data["parameters"]["angleTwo"]
    assert angle_one != 0 or angle_two != 0, (
        "At least one of angleOne or angleTwo must be non-zero"
    )
    axis_point = revolve_data["parameters"]["axis"]["point"]
    axis_dir = revolve_data["parameters"]["axis"]["direction"]
    cq_commands = f"{shape_name} = {wp_name}.revolve_bidir(angleDegrees=({angle_one}, {angle_two}), axisPoint={tuple(axis_point)}, axisDir={tuple(axis_dir)})"
    return cq_commands


def _refine2cq(refine_data, shape_name, com=True, query_idx=None):
    if com:
        assert query_idx is None, (
            "Cannot use both CoM and query_idx in refine operation"
        )
    op_name = refine_data["type"]
    cq_code = ""
    if "radius" in refine_data["parameters"]:
        dimensions = refine_data["parameters"]["radius"]
    elif "width" in refine_data["parameters"]:
        dimensions = refine_data["parameters"]["width"]
    elif "thickness" in refine_data["parameters"]:
        dimensions = -1 * refine_data["parameters"]["thickness"]
    else:
        raise NotImplementedError(
            f"Refine operation {op_name} missing dimension parameter"
        )
    if op_name == "shell":
        dimensions = f'{dimensions}, kind="intersection"'
    if com:
        refine_data["CoM"]["edges"] = [
            tuple(edge) for edge in refine_data["CoM"]["edges"]
        ]
        refine_data["CoM"]["faces"] = [
            tuple(face) for face in refine_data["CoM"]["faces"]
        ]
        assert (
            len(refine_data["CoM"]["edges"]) + len(refine_data["CoM"]["faces"]) > 0
        ), "No entities to refine"
        centers_edge_str = (
            "[" + ", ".join(str(e) for e in refine_data["CoM"]["edges"]) + "]"
        )
        centers_face_str = (
            "[" + ", ".join(str(f) for f in refine_data["CoM"]["faces"]) + "]"
        )
        selector_edge_str = f"cq.selectors.NearestToPointsSelector({centers_edge_str})"
        selector_face_str = f"cq.selectors.NearestToPointsSelector({centers_face_str})"
        if len(refine_data["CoM"]["edges"]) > 0:
            cq_code += f"{shape_name} = {shape_name}.edges({selector_edge_str}).{op_name}({dimensions})"
            cq_code += "\n"
        if len(refine_data["CoM"]["faces"]) > 0:
            cq_code += f"{shape_name} = {shape_name}.faces({selector_face_str}).{op_name}({dimensions})"
            cq_code += "\n"
    elif query_idx is not None:
        query_str = refine_data["query"][query_idx]
        cq_code += f"{shape_name} = {shape_name}.{op_name}Q(cq.Query({query_str!r}), {dimensions})"
        cq_code += "\n"
    return cq_code


def _generate_code_with_refine(json_data, com=True, query_idx=None):
    cq_code = ""
    sketch_count = 0
    sketch_prefix = "sketch"
    shape_count = 0
    shape_prefix = "shape"
    first_shape = "model"
    for seq in json_data["sequence"]:
        if seq["type"] in ["sketch"]:
            continue
        if seq["type"] in ["extrude", "revolve"]:
            sketch_op = json_data["features"][seq["feature_id"]]
            sketch = json_data["features"][sketch_op["sketch_id"]]
            sketch_name = f"{sketch_prefix}{sketch_count}"
            sketch_count += 1
            cq_code += _workplane_cq(sketch["plane"], sketch_name)
            cq_code += "\n"
            for face in sketch["profiles"].values():
                loops = face["loops"]
                for loop in loops:
                    loop = loop["loop_curves"]
                    cq_code += _loop2cq(loop, sketch_name)
                    cq_code += "\n"
            shape_name = (
                first_shape if shape_count == 0 else f"{shape_prefix}{shape_count}"
            )
            shape_count += 1
            if sketch_op["type"] == "extrude":
                cq_code += _extrude2cq(sketch_op, sketch_name, shape_name)
                cq_code += "\n"
            elif sketch_op["type"] == "revolve":
                cq_code += _revolve2cq(sketch_op, sketch_name, shape_name)
                cq_code += "\n"
            else:
                raise NotImplementedError(
                    f"Feature type {sketch_op['type']} not implemented"
                )
            if shape_name != first_shape:
                bool_op = sketch_op["parameters"]["operationType"]
                if bool_op in ["NEW", "ADD"]:
                    bool_op = "union"
                elif bool_op == "REMOVE":
                    bool_op = "cut"
                elif bool_op == "INTERSECT":
                    bool_op = "intersect"
                else:
                    raise NotImplementedError(
                        f"Boolean operation {bool_op} not implemented"
                    )
                cq_code += f"{first_shape} = {first_shape}.{bool_op}({shape_name})"
                cq_code += "\n"
            cq_code += "\n"
        else:
            refine_op = json_data["features"][seq["feature_id"]]
            cq_code += _refine2cq(refine_op, first_shape, com=com, query_idx=query_idx)
            cq_code += "\n"
    return cq_code
