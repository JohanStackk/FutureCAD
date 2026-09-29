import numpy as np
from scipy.spatial import cKDTree


def validate_points(points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a nonempty N x 3 point cloud")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite values")
    return points


def normalize_pc(points):
    points = validate_points(points)
    centered = points - np.mean(points, axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale <= 0:
        raise ValueError("Cannot normalize a zero-radius point cloud")
    return centered / scale


def chamfer_dist(gt_points, gen_points):
    gt = validate_points(gt_points)
    gen = validate_points(gen_points)
    a = cKDTree(gen).query(gt)[0]
    b = cKDTree(gt).query(gen)[0]
    return float(np.mean(a**2) + np.mean(b**2))


def reward_linear_cd(cd, t=0.2):
    if not np.isfinite(cd) or not np.isfinite(t) or t <= 0:
        raise ValueError("CD must be finite and reward tolerance must be positive")
    return float(1 - np.clip(cd, 0, t) / t)


def read_ply(path):
    from plyfile import PlyData

    vertices = PlyData.read(str(path))["vertex"]
    return validate_points(np.stack([vertices[k] for k in ("x", "y", "z")], axis=1))


def write_ply(points, path):
    from plyfile import PlyData, PlyElement

    points = normalize_pc(points)
    vertices = np.array(
        [tuple(p) for p in points], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def stl2pc(path, n_points=2048, seed=None):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    if mesh.is_empty or mesh.area <= 0:
        raise ValueError("Cannot sample an empty or zero-area mesh")
    if seed is None:
        return trimesh.sample.sample_surface(mesh, n_points)[0]
    return trimesh.sample.sample_surface(mesh, n_points, seed=seed)[0]


def lower_median(values):
    values = np.asarray(values)
    if not len(values):
        return None
    k = (len(values) - 1) // 2
    return float(np.partition(values, k)[k])
