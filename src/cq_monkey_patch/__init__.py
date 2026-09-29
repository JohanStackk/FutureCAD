import cadquery as cq

from .workplane_ext import WorkplaneEx, set_filter_in_refine
from .plane_ext import PlaneEx
from .query import Query
from .selector import NearestToPointsSelector

__all__ = ["Query", "set_filter_in_refine"]

cq.selectors.NearestToPointsSelector = NearestToPointsSelector
cq.Workplane = WorkplaneEx
cq.Query = Query
cq.Plane.named = PlaneEx.named
