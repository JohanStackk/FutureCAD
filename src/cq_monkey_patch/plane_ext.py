from typing import Type

from cadquery import Plane


class PlaneEx(Plane):
    @classmethod
    def named(cls: Type["Plane"], stdName: str, origin=(0, 0, 0)) -> "Plane":
        namedPlanes = {
            "XY": Plane(origin, (1, 0, 0), (0, 0, 1)),
            "-XY": Plane(origin, (1, 0, 0), (0, 0, -1)),
            "YZ": Plane(origin, (0, 1, 0), (1, 0, 0)),
            "-YZ": Plane(origin, (0, 1, 0), (-1, 0, 0)),
            "ZX": Plane(origin, (0, 0, 1), (0, 1, 0)),
            "-ZX": Plane(origin, (0, 0, 1), (0, -1, 0)),
            "XZ": Plane(origin, (1, 0, 0), (0, -1, 0)),
            "-XZ": Plane(origin, (1, 0, 0), (0, 1, 0)),
            "YX": Plane(origin, (0, 1, 0), (0, 0, -1)),
            "-YX": Plane(origin, (0, 1, 0), (0, 0, 1)),
            "ZY": Plane(origin, (0, 0, 1), (-1, 0, 0)),
            "-ZY": Plane(origin, (0, 0, 1), (1, 0, 0)),
            "front": Plane(origin, (1, 0, 0), (0, 0, 1)),
            "back": Plane(origin, (-1, 0, 0), (0, 0, -1)),
            "left": Plane(origin, (0, 0, 1), (-1, 0, 0)),
            "right": Plane(origin, (0, 0, -1), (1, 0, 0)),
            "top": Plane(origin, (1, 0, 0), (0, 1, 0)),
            "bottom": Plane(origin, (1, 0, 0), (0, -1, 0)),
        }

        try:
            return namedPlanes[stdName]
        except KeyError:
            raise ValueError("Supported names are {}".format(list(namedPlanes.keys())))
