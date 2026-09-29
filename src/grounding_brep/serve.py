import argparse
import threading
import torch
import dgl
from flask import Flask, request, jsonify
from grounding_brep.runtime import load_config, load_model
from futurecad.grounding_threshold import resolve_threshold


def create_app(model, device, threshold):
    app = Flask(__name__)
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return jsonify(status="ready", inference_threshold=threshold)

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json()
            data = payload["graph"]
            g = dgl.graph(
                (data["edges"]["src"], data["edges"]["dst"]),
                num_nodes=data["num_nodes"],
            )
            for key, value in data["ndata"].items():
                g.ndata[key] = torch.tensor(value, dtype=torch.float32)
            for key, value in data["edata"].items():
                g.edata[key] = torch.tensor(value, dtype=torch.float32)
            if g.num_edges() % 2:
                raise ValueError("Expected adjacent pairs of directed edges")
            with lock, torch.inference_mode():
                batch = {
                    "graph": dgl.batch([g]).to(device),
                    "query": [payload["query"]],
                }
                probs = torch.sigmoid(model(batch)["pred_scores"][0, :, 0])
                labels = (probs >= threshold).int().cpu().tolist()
            return jsonify(success=True, labels=labels, num_elements=len(labels))
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(success=False, error=str(error)), 400

    return app


def main():
    parser = argparse.ArgumentParser('Local BRepGround inference endpoint used by cq.Query.')
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--allow-threshold-override", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    model = load_model(cfg, args.checkpoint, args.device).eval()
    declared = getattr(model, "checkpoint_inference_threshold", None)
    threshold = resolve_threshold(
        cfg.model.inference.threshold,
        declared,
        allow_override=args.allow_threshold_override,
    )
    print(
        f"BRepGround inference threshold: {threshold:g}; checkpoint declaration: {declared}",
        flush=True,
    )
    create_app(model, args.device, threshold).run(
        host="127.0.0.1", port=args.port, threaded=True
    )


if __name__ == "__main__":
    main()
