import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from futurecad.splits import split_ids


class GroundingBRepDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        limit=None,
        max_len_uv=None,
        split_path=None,
    ):
        self.root_dir = Path(root_dir)
        if split_path is None:
            raise ValueError("Supply the final split file through data.split_path")
        files = split_ids(split_path, split)
        if limit is not None and limit > 0:
            files = files[:limit]
        self.samples = []
        self.graph_dir, self.label_dir = (
            self.root_dir / "graph",
            self.root_dir / "labels",
        )
        for name in files:
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Invalid sample name: {name!r}")
            graph_path = self.graph_dir / f"{name}.bin"
            if not graph_path.is_file():
                raise FileNotFoundError(graph_path)
            data = json.loads(
                (self.label_dir / f"{name}.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(data["query"], list)
                or not data["query"]
                or any(
                    not isinstance(query, str) or not query.strip()
                    for query in data["query"]
                )
            ):
                raise ValueError(
                    f"Expected a nonempty list of query strings for {name}"
                )
            length = len(data["labels"])
            if max_len_uv is not None and length > max_len_uv:
                continue
            if not set(data["labels"]).issubset({0, 1}):
                raise ValueError(f"Non-binary labels for {name}")
            for q in range(len(data["query"])):
                self.samples.append((name, q, length))
        self.samples.sort(key=lambda x: (x[2], x[0], x[1]))
        self.samples = [(name, q) for name, q, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from dgl.data.utils import load_graphs

        name, q = self.samples[idx]
        graphs, _ = load_graphs(str(self.graph_dir / f"{name}.bin"))
        graph = graphs[0]
        for fields in (graph.ndata, graph.edata):
            for k in fields:
                fields[k] = fields[k].float()
        data = json.loads((self.label_dir / f"{name}.json").read_text(encoding="utf-8"))
        if (
            graph.num_edges() % 2
            or len(data["labels"]) != graph.num_nodes() + graph.num_edges() // 2
        ):
            raise ValueError(f"Graph/label dimensions disagree for {name}")
        return {
            "graph": graph,
            "query": data["query"][q],
            "labels": {"labels": torch.tensor(data["labels"], dtype=torch.long)},
            "filename": name,
        }

    @staticmethod
    def collate_fn(samples):
        import dgl

        return {
            "graph": dgl.batch([s["graph"] for s in samples]),
            "query": [s["query"] for s in samples],
            "filename": [s["filename"] for s in samples],
            "labels": {
                "labels": torch.nn.utils.rnn.pad_sequence(
                    [s["labels"]["labels"] for s in samples],
                    batch_first=True,
                    padding_value=-1,
                )
            },
        }

    def get_dataloader(
        self, batch_size, shuffle, num_workers, drop_last=False, pin_memory=True
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
