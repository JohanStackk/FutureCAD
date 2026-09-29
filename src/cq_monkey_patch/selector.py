from typing import Sequence

from cadquery import Selector


class NearestToPointsSelector(Selector):
    def __init__(self, pnts: list):
        self.pnts = pnts

    def filter(self, objectList: Sequence):
        selected = set()
        for p in self.pnts:
            px, py, pz = p

            def dist(shp):
                center = shp.Center()
                return (
                    (center.x - px) ** 2 + (center.y - py) ** 2 + (center.z - pz) ** 2
                )

            nearest = min(objectList, key=dist)
            selected.add(nearest)
        return [s for s in selected]
