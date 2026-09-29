from typing import Union, Literal, Optional, Tuple, cast, List

import cadquery as cq
from cadquery import Face, Solid, Compound, Vector, Edge
from cadquery.cq import T, CombineMode, VectorLike

from occwl.graph import bbox_hash
from .query import Query

FILTER_IN_REFINE = True


def set_filter_in_refine(value: bool):
    global FILTER_IN_REFINE
    FILTER_IN_REFINE = value


class WorkplaneEx(cq.Workplane):
    def extrude_bidir(
        self: T,
        distance: Tuple[float, float],
        combine: CombineMode = True,
        clean: bool = True,
        both: bool = False,
        taper: Optional[float] = None,
    ) -> T:
        until = distance

        r = self._extrude_bidir(
            until, both=both, taper=taper, upToFace=None, both_differ=True
        )

        return self._combineWithBase(r, combine, clean)

    def _extrude_bidir(
        self,
        distance: Union[float, float] = None,
        both: bool = False,
        taper: Optional[float] = None,
        upToFace: Optional[Union[int, Face]] = None,
        additive: bool = True,
        both_differ=False,
    ) -> Union[Solid, Compound]:
        def getFacesList(face, eDir, direction, both=False):
            facesList = self.findSolid().facesIntersectedByLine(
                face.Center(), eDir, direction=direction
            )
            if len(facesList) == 0 and both:
                raise ValueError(
                    "Couldn't find a face to extrude/cut to for at least one of the two required directions of extrusion/cut."
                )

            if len(facesList) == 0:
                facesList = self.findSolid().facesIntersectedByLine(
                    face.Center(), eDir.multiply(-1.0), direction=direction
                )
                if len(facesList) == 0:
                    raise ValueError(
                        "Couldn't find a face to extrude/cut to. Check your workplane orientation."
                    )
            return facesList

        if both_differ:
            temp = distance[0]
            distance_two = distance[1]
            distance = temp
        faces = self._getFaces()

        for face in faces:
            if taper and face.innerWires():
                raise ValueError("Inner wires not allowed with tapered extrusion")

        if upToFace is not None:
            eDir = self.plane.zDir
        elif distance is not None:
            eDir = self.plane.zDir.multiply(distance)
            eDir_two = self.plane.zDir.multiply(-1 * distance_two)

        direction = "AlongAxis" if additive else "Opposite"
        taper = 0.0 if taper is None else taper

        if upToFace is not None:
            res = self.findSolid()
            for face in faces:
                if isinstance(upToFace, int):
                    facesList = getFacesList(face, eDir, direction, both=both)
                    if (
                        res.isInside(face.outerWire().Center())
                        and additive
                        and upToFace == 0
                    ):
                        upToFace = 1

                    limitFace = facesList[upToFace]
                else:
                    limitFace = upToFace

                res = res.dprism(
                    None,
                    [face],
                    taper=taper,
                    upToFace=limitFace,
                    additive=additive,
                )

                if both:
                    facesList2 = getFacesList(
                        face, eDir.multiply(-1.0), direction, both=both
                    )
                    limitFace2 = facesList2[upToFace]
                    res = res.dprism(
                        None,
                        [face],
                        taper=taper,
                        upToFace=limitFace2,
                        additive=additive,
                    )
        else:
            toFuse = []
            for face in faces:
                if distance != 0:
                    s1 = Solid.extrudeLinear(face, eDir, taper=taper)
                    toFuse.append(s1)

                if distance_two != 0:
                    s2 = Solid.extrudeLinear(face, eDir_two, taper=taper)
                    toFuse.append(s2)

            res = Compound.makeCompound(toFuse)
        return res

    def revolve_bidir(
        self: T,
        angleDegrees: Tuple[float, float] = 360.0,
        axisPoint: Tuple[float, float, float] = None,
        axisDir: Tuple[float, float, float] = None,
        combine: CombineMode = True,
        clean: bool = True,
    ) -> T:
        axisEnd = (
            axisPoint[0] + axisDir[0],
            axisPoint[1] + axisDir[1],
            axisPoint[2] + axisDir[2],
        )
        r = self._revolve_bidir(angleDegrees, axisPoint, axisEnd)

        return self._combineWithBase(r, combine, clean)

    def _revolve_bidir(
        self,
        angleDegrees: Tuple[float, float],
        axisStart: VectorLike,
        axisEnd: VectorLike,
    ) -> Compound:
        toFuse = []
        for f in self._getFaces():
            if angleDegrees[0] != 0:
                thisObj1 = Solid.revolve(
                    f, angleDegrees[0], Vector(axisStart), Vector(axisEnd)
                )
                toFuse.append(thisObj1)

            if angleDegrees[1] != 0:
                thisObj2 = Solid.revolve(
                    f, -angleDegrees[1], Vector(axisStart), Vector(axisEnd)
                )
                toFuse.append(thisObj2)

        return Compound.makeCompound(toFuse)

    @staticmethod
    def get_bbox(obj):
        bbox = obj.BoundingBox()
        return {
            "min": [bbox.xmin, bbox.ymin, bbox.zmin],
            "max": [bbox.xmax, bbox.ymax, bbox.zmax],
        }

    def _processQuery(self, solid, query: Query):
        t_face_bboxes, t_edge_bboxes = query.query(solid.wrapped)

        faces = self.faces().objects
        edges = self.edges().objects

        target_face_set = set(bbox_hash(tb, digits=4) for tb in t_face_bboxes)
        target_edge_set = set(bbox_hash(tb, digits=4) for tb in t_edge_bboxes)

        faceList = []
        edgeList = []

        for face in faces:
            face_bbox = bbox_hash(WorkplaneEx.get_bbox(face))
            if face_bbox in target_face_set:
                faceList.append(face)
        for edge in edges:
            edge_bbox = bbox_hash(WorkplaneEx.get_bbox(edge))
            if edge_bbox in target_edge_set:
                edgeList.append(edge)

        return faceList, edgeList

    def filletQ(self: T, query: Query, radius: float) -> T:
        solid = self.findSolid()

        faceList, edgeList = self._processQuery(solid, query)

        for face in faceList:
            edgeList.extend(face.Edges())

        if len(edgeList) < 1:
            raise ValueError("Fillets requires that edges be selected")

        if not FILTER_IN_REFINE:
            s = solid.fillet(radius, edgeList)
            return self.newObject([s.clean()])

        try:
            s = solid.fillet(radius, edgeList)
        except:
            valid_edges = []
            for edge in edgeList:
                try:
                    solid.fillet(radius, [edge])
                    valid_edges.append(edge)
                except:
                    pass
            if len(valid_edges) < 1:
                raise ValueError(
                    "No valid edges could be filleted with the given radius"
                )
            s = solid.fillet(radius, valid_edges)
            print(
                f"Warning: Only {len(valid_edges)} out of {len(edgeList)} edges could be filleted."
            )

        return self.newObject([s.clean()])

    def chamferQ(self: T, query: Query, length: float) -> T:
        solid = self.findSolid()

        faceList, edgeList = self._processQuery(solid, query)

        for face in faceList:
            edgeList.extend(face.Edges())

        if len(edgeList) < 1:
            raise ValueError("Chamfer requires that edges be selected")

        if not FILTER_IN_REFINE:
            s = solid.chamfer(length, None, edgeList)
            return self.newObject([s.clean()])

        try:
            s = solid.chamfer(length, None, edgeList)
        except:
            valid_edges = []
            for edge in edgeList:
                try:
                    solid.chamfer(length, None, [edge])
                    valid_edges.append(edge)
                except:
                    pass
            if len(valid_edges) < 1:
                raise ValueError(
                    "No valid edges could be chamfered with the given length"
                )
            s = solid.chamfer(length, None, valid_edges)
            print(
                f"Warning: Only {len(valid_edges)} out of {len(edgeList)} edges could be chamfered."
            )

        return self.newObject([s.clean()])

    def shellQ(
        self: T,
        query: Query,
        thickness: float,
        kind: Literal["arc", "intersection"] = "intersection",
    ) -> T:
        solid = self.findSolid()

        faces, _ = self._processQuery(solid, query)

        if len(faces) < 1:
            raise ValueError("Shell requires that faces be selected")

        if not FILTER_IN_REFINE:
            s = solid.shell(faces, thickness, kind=kind)
            return self.newObject([s.clean()])

        try:
            s = solid.shell(faces, thickness, kind=kind)
        except:
            valid_faces = []
            for face in faces:
                try:
                    solid.shell([face], thickness, kind=kind)
                    valid_faces.append(face)
                except:
                    pass
            if len(valid_faces) < 1:
                raise ValueError(
                    "No valid faces could be shelled with the given thickness"
                )
            s = solid.shell(valid_faces, thickness, kind=kind)
            print(
                f"Warning: Only {len(valid_faces)} out of {len(faces)} faces could be shelled."
            )
        return self.newObject([s])

    def fillet(self: T, radius: float) -> T:
        solid = self.findSolid()

        edgeList = cast(List[Edge], self.edges().vals())
        if len(edgeList) < 1:
            raise ValueError("Fillets requires that edges be selected")

        if not FILTER_IN_REFINE:
            s = solid.fillet(radius, edgeList)
            return self.newObject([s.clean()])

        try:
            s = solid.fillet(radius, edgeList)
        except:
            valid_edges = []
            for edge in edgeList:
                try:
                    solid.fillet(radius, [edge])
                    valid_edges.append(edge)
                except:
                    pass
            if len(valid_edges) < 1:
                raise ValueError(
                    "No valid edges could be filleted with the given radius"
                )
            s = solid.fillet(radius, valid_edges)
            print(
                f"Warning: Only {len(valid_edges)} out of {len(edgeList)} edges could be filleted."
            )
        return self.newObject([s.clean()])

    def chamfer(self: T, length: float, length2: Optional[float] = None) -> T:
        solid = self.findSolid()

        edgeList = cast(List[Edge], self.edges().vals())
        if len(edgeList) < 1:
            raise ValueError("Chamfer requires that edges be selected")

        if not FILTER_IN_REFINE:
            s = solid.chamfer(length, length2, edgeList)
            return self.newObject([s])

        try:
            s = solid.chamfer(length, None, edgeList)
        except:
            valid_edges = []
            for edge in edgeList:
                try:
                    solid.chamfer(length, None, [edge])
                    valid_edges.append(edge)
                except:
                    pass
            if len(valid_edges) < 1:
                raise ValueError(
                    "No valid edges could be chamfered with the given length"
                )
            s = solid.chamfer(length, None, valid_edges)
            print(
                f"Warning: Only {len(valid_edges)} out of {len(edgeList)} edges could be chamfered."
            )

        return self.newObject([s.clean()])

    def shell(
        self: T, thickness: float, kind: Literal["arc", "intersection"] = "intersection"
    ) -> T:
        solid = self.findSolid()

        faces = [f for f in self.objects if isinstance(f, Face)]

        if not FILTER_IN_REFINE:
            s = solid.shell(faces, thickness, kind=kind)
            return self.newObject([s])

        try:
            s = solid.shell(faces, thickness, kind=kind)
        except:
            valid_faces = []
            for face in faces:
                try:
                    solid.shell([face], thickness, kind=kind)
                    valid_faces.append(face)
                except:
                    pass
            if len(valid_faces) < 1:
                raise ValueError(
                    "No valid faces could be shelled with the given thickness"
                )
            s = solid.shell(valid_faces, thickness, kind=kind)
            print(
                f"Warning: Only {len(valid_faces)} out of {len(faces)} faces could be shelled."
            )
        return self.newObject([s])
