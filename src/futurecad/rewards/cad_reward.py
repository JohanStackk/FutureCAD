from pathlib import Path
import re
import subprocess
import tempfile

from futurecad.execution import execute_to_stl
from futurecad.geometry import (
    chamfer_dist,
    normalize_pc,
    read_ply,
    reward_linear_cd,
    stl2pc,
    write_ply,
)

CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)
FORMAT_WEIGHT = 0.1
GEOMETRY_WEIGHT = 0.9


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    matches = CODE_PATTERN.findall(solution_str)
    if not matches:
        return 0.0
    gt_pc = read_ply(ground_truth["ply_path"])
    with tempfile.TemporaryDirectory(prefix="futurecad-reward-") as tmp:
        stl_path, ply_path = Path(tmp) / "model.stl", Path(tmp) / "model.ply"
        try:
            execute_to_stl(matches[-1].strip(), stl_path)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired):
            return 0.0
        pc = normalize_pc(stl2pc(stl_path, 2048))
        write_ply(pc, ply_path)
        cd = chamfer_dist(gt_pc, read_ply(ply_path))
        return FORMAT_WEIGHT + GEOMETRY_WEIGHT * reward_linear_cd(cd)
