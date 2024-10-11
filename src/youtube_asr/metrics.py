from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torchmetrics import Metric

if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    Phase: TypeAlias = Literal["train", "val", "test"]


class MultiLabelMetric(Metric):
    """My custom metric to compute classification metrics.

    Computes precision, recall, f1, and support for all labels, plus micro and
    macro averages.
    """

    def __init__(
        self, target_names: list[str], threshold: float = 0.5, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("tp", torch.zeros((len(target_names),)), dist_reduce_fx="sum")
        self.add_state("fp", torch.zeros((len(target_names),)), dist_reduce_fx="sum")
        self.add_state("fn", torch.zeros((len(target_names),)), dist_reduce_fx="sum")
        self.add_state(
            "support", torch.zeros((len(target_names),)), dist_reduce_fx="sum"
        )
        self.tp: torch.Tensor
        self.fp: torch.Tensor
        self.fn: torch.Tensor
        self.support
        self.target_names = target_names
        self.threshold = threshold

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (torch.nn.functional.sigmoid(logits) > self.threshold).to(torch.int32)

        self.support += targets.sum(dim=0)

        self.tp += (preds * targets).sum(dim=0)
        self.fp += (preds * (1 - targets)).sum(dim=0)
        self.fn += ((1 - preds) * targets).sum(dim=0)

    @staticmethod
    def _compute_precision_recall_f1(
        tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Invalid when tp + fp == 0 or tp + fn == 0
        invalid_mask = (((tp + fp) * (tp + fn)) == 0).to(torch.int32)

        precision = (1 - invalid_mask) * tp / (tp + fp + invalid_mask)
        recall = (1 - invalid_mask) * tp / (tp + fn + invalid_mask)

        f1_denom = 1 / (precision + invalid_mask) + 1 / (recall + invalid_mask)
        f1 = (1 - invalid_mask) * 2 / f1_denom
        return (invalid_mask, precision, recall, f1)

    @torch.inference_mode()
    def compute(self) -> dict[str, torch.Tensor]:
        invalid_mask, *metrics = self._compute_precision_recall_f1(
            self.tp, self.fp, self.fn
        )

        metric_names = ["precision", "recall", "f1"]

        per_label_metrics = {
            f"{metric_name}/{label}": metric[i]
            for i, label in enumerate(self.target_names)
            for metric_name, metric in zip(metric_names, metrics, strict=True)
        }
        supports = {
            f"support/{label}": self.support[i]
            for i, label in enumerate(self.target_names)
        }
        supports["support"] = self.support.sum()

        valid_targets = (1 - invalid_mask).sum()
        macro_averaged = {
            f"{metric_name}/macro": metric.sum() / valid_targets
            if valid_targets > 0
            else torch.tensor(0)
            for metric_name, metric in zip(metric_names, metrics, strict=True)
        }

        all_tps = self.tp.sum()
        all_fps = self.fp.sum()
        all_fns = self.fn.sum()

        invalid_mask, *micro_metrics = self._compute_precision_recall_f1(
            all_tps, all_fps, all_fns
        )

        micro_averaged = {
            f"{metric_name}/micro": metric
            for metric_name, metric in zip(metric_names, micro_metrics, strict=True)
        }

        return per_label_metrics | macro_averaged | micro_averaged | supports
