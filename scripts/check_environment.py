import argparse
import importlib.metadata
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Check the installed CAD bridge, GPU graph operators, or LLM runtime.')
    parser.add_argument(
        "--component", choices=("all", "cad", "grounding", "llm"), default="all"
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.component == "all":
        records = []
        for component in ("cad", "grounding", "llm"):
            result = subprocess.run(
                [sys.executable, __file__, "--component", component, "--device", args.device],
                capture_output=True, text=True,
            )
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            records.append({"component": component, "passed": result.returncode == 0})
        if args.output:
            from pathlib import Path

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
        if not all(record["passed"] for record in records):
            raise SystemExit(1)
        return
    report = {"component": args.component, "python": sys.version, "status": "passed"}
    try:
        if args.component == "cad":
            import cadquery as cq
            import cq_monkey_patch
            from cq_monkey_patch.query import ocp_to_pythonocc
            from cq_monkey_patch._brep_to_graph import build_graph_as_dict
            from occwl.solid import Solid
            import occwl

            assert cq_monkey_patch.Query is cq.Query
            box = cq.Workplane("XY").box(1, 2, 3)
            solid = Solid(ocp_to_pythonocc(box.val().wrapped), allow_compound=True)
            graph, faces, edges = build_graph_as_dict(solid)
            assert graph["num_nodes"] == len(faces) and len(edges) > 0
            assert hasattr(cq, "Query")
            report.update(occwl=occwl.__file__, faces=len(faces), edges=len(edges))
            packages = ["cadquery", "cadquery-ocp", "numpy"]
        elif args.component == "grounding":
            import torch
            import dgl
            import dgl.function as fn
            from grounding_brep.models.grounding import GroundingBRep
            from transformers import BertModel

            assert GroundingBRep and BertModel
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            graph = dgl.graph(([0, 1], [1, 0]), device=args.device)
            graph.ndata["x"] = torch.ones(2, 1, device=args.device)
            graph.update_all(fn.copy_u("x", "m"), fn.sum("m", "y"))
            assert torch.equal(graph.ndata["y"], graph.ndata["x"])
            from dgl.nn import NNConv

            layer = NNConv(4, 4, torch.nn.Linear(2, 16)).to(args.device)
            nodes = torch.ones(2, 4, device=args.device, requires_grad=True)
            edges = torch.ones(2, 2, device=args.device)
            layer(graph, nodes, edges).sum().backward()
            assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
            report.update(device=args.device, torch_cuda=torch.version.cuda)
            if args.device == "cuda":
                report["gpu"] = torch.cuda.get_device_name()
            packages = ["torch", "dgl", "lightning", "transformers"]
        else:
            import torch
            from vllm import LLM

            assert LLM
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            report["gpu"] = torch.cuda.get_device_name()
            report["scope"] = "Import/CUDA check only; does not load the LLM checkpoint"
            packages = ["torch", "vllm", "transformers"]
        report["packages"] = {
            name: importlib.metadata.version(name) for name in packages
        }
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        from pathlib import Path

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
