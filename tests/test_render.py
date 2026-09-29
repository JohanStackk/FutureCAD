import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from futurecad.execution import execute_to_files


@unittest.skipUnless(importlib.util.find_spec("cadquery"), "CadQuery is required")
class RenderTests(unittest.TestCase):
    def test_svg_and_solid_exports(self):
        code = "import cadquery as cq\nmodel = cq.Workplane('XY').box(2, 3, 4)\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {kind: root / f"model.{kind.lower()}" for kind in ("STEP", "STL", "SVG")}
            execute_to_files(code, outputs)
            svg = ET.parse(outputs["SVG"]).getroot()
            self.assertTrue(svg.tag.endswith("svg"))
            paths = svg.findall(".//{http://www.w3.org/2000/svg}path")
            self.assertTrue(any(path.get("d") for path in paths))
            self.assertGreater(outputs["STEP"].stat().st_size, 100)
            self.assertGreater(outputs["STL"].stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
