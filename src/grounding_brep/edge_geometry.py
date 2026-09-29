import math

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_REVERSED

EPS_VEC = 1e-12
EPS_LIN = 1e-7
EPS_ANG = 3


def _edge_mid_point_tangent(edge):
    curve = BRepAdaptor_Curve(edge)
    f = curve.FirstParameter()
    l = curve.LastParameter()
    t = 0.5 * (f + l)

    P = curve.Value(t)
    V = gp_Vec()
    curve.D1(t, P, V)

    if V.Magnitude() <= EPS_VEC:
        raise ValueError("Degenerate edge")

    if edge.Orientation() == TopAbs_REVERSED:
        V.Reverse()

    V.Normalize()
    return t, P, V


def _face_normal_at_edge_param(edge, face, t_edge):
    curve = BRepAdaptor_Curve(edge)
    P = curve.Value(t_edge)
    surf_handle = BRep_Tool.Surface(face)
    sas = ShapeAnalysis_Surface(surf_handle)
    uv = sas.ValueOfUV(P, EPS_LIN)
    u, v = uv.X(), uv.Y()
    surf_adp = BRepAdaptor_Surface(face)

    Ptmp = curve.Value(t_edge)
    dU = gp_Vec()
    dV = gp_Vec()

    surf_adp.D1(u, v, Ptmp, dU, dV)
    N = dU.Crossed(dV)
    if N.Magnitude() <= EPS_VEC:
        raise ValueError("Degenerate surface normal")

    if face.Orientation() == TopAbs_REVERSED:
        N.Reverse()

    N.Normalize()
    return N


def normal_angle_deg(N1, N2):
    sin_theta = N1.Crossed(N2).Magnitude()
    cos_theta = max(-1.0, min(1.0, N1.Dot(N2)))
    theta = math.atan2(sin_theta, cos_theta)
    return math.degrees(theta)


def edge_convexity(edge, face1, face2, solid):
    t, P, T = _edge_mid_point_tangent(edge)

    N1 = _face_normal_at_edge_param(edge, face1, t)
    N2 = _face_normal_at_edge_param(edge, face2, t)
    cross = N1.Crossed(N2)
    angle = normal_angle_deg(N1, N2)
    if angle <= EPS_ANG:
        return "smooth"

    s = cross.Dot(T)

    W = gp_Vec(N1.X() + N2.X(), N1.Y() + N2.Y(), N1.Z() + N2.Z())

    if W.Magnitude() <= EPS_VEC:
        W = cross.Crossed(T)
        if W.Magnitude() <= EPS_VEC:
            return "convex" if s > 0 else "concave"

    W.Normalize()
    Ptest = gp_Pnt(
        P.X() + W.X() * EPS_LIN * 10,
        P.Y() + W.Y() * EPS_LIN * 10,
        P.Z() + W.Z() * EPS_LIN * 10,
    )

    clf = BRepClass3d_SolidClassifier(solid)
    clf.Perform(Ptest, EPS_LIN)

    if clf.State() in (TopAbs_IN, TopAbs_ON):
        s = -s

    return "convex" if s > 0 else "concave"
