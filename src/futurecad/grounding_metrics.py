import numpy as np


class GroundingMetrics:
    def __init__(self, threshold=0.5, ks=(3, 5, 10)):
        self.threshold, self.ks = threshold, tuple(ks)
        self.tp = self.fp = self.fn = self.tn = self.queries = 0
        self.ap = 0.0
        self.recalls = {k: 0.0 for k in self.ks}

    def update(self, probs, labels):
        for scores, targets in zip(probs, labels, strict=True):
            scores, targets = np.asarray(scores), np.asarray(targets)
            if scores.shape != targets.shape or scores.ndim != 1:
                raise ValueError(
                    "Probabilities and labels must have matching one-dimensional shapes"
                )
            if not np.isin(targets, [-1, 0, 1]).all():
                raise ValueError("Expected binary labels or -1 padding")
            mask = targets != -1
            scores, targets = np.asarray(scores)[mask], np.asarray(targets)[mask]
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite model probabilities")
            pred = scores >= self.threshold
            truth = targets == 1
            self.tp += int(np.sum(pred & truth))
            self.fp += int(np.sum(pred & ~truth))
            self.fn += int(np.sum(~pred & truth))
            self.tn += int(np.sum(~pred & ~truth))
            n = int(truth.sum())
            if n == 0:
                continue
            order = np.argsort(-scores, kind="stable")
            ranked = truth[order]
            self.ap += float(
                np.sum((np.cumsum(ranked) / np.arange(1, len(ranked) + 1))[ranked]) / n
            )
            self.queries += 1
            for k in self.ks:
                self.recalls[k] += float(ranked[:k].sum() / n)

    def result(self):
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "ranking_queries": self.queries,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "mAP": self.ap / self.queries if self.queries else None,
            **{
                f"recall@{k}": v / self.queries if self.queries else None
                for k, v in self.recalls.items()
            },
        }
