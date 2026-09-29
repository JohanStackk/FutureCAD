import ast

COMMANDS = {
    "line": "lineTo",
    "arc": "threePointArc",
    "circle": "circle",
    "spline": "spline",
    "extrude": ".extrude",
    "revolve": ".revolve",
    "fillet": ".fillet",
    "chamfer": ".chamfer",
    "shell": ".shell",
}
GROUPS = {
    "curve_f1": ("line", "arc", "circle", "spline"),
    "3d_op_f1": ("extrude", "revolve"),
    "refine_f1": ("fillet", "chamfer", "shell"),
}


def count_f1(gt, pred):
    return 2 * min(gt, pred) / (gt + pred) if min(gt, pred) > 0 else -1.0


def program_f1(code, gt_counts):
    tree = ast.parse(code)
    if any(s in code for s in ("cq_gears", "mirror", "rotate", "array")):
        return None
    if any(
        isinstance(n, (ast.For, ast.FunctionDef, ast.AsyncFunctionDef))
        for n in ast.walk(tree)
    ):
        return None
    pred = {key: code.count(token) for key, token in COMMANDS.items()}
    return {
        group: count_f1(sum(gt_counts[k] for k in keys), sum(pred[k] for k in keys))
        for group, keys in GROUPS.items()
    }
