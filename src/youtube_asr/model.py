from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import Metric

from youtube_asr.metrics import MultiLabelMetric

if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    Phase: TypeAlias = Literal["train", "val", "test"]


class ResNetBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        layers = [
            torch.nn.Conv2d(channels, channels, kernel, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel, padding="same"),
        ]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        return outputs + inputs


class PreActivationResNetBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        layers = [
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel, padding="same"),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel, padding="same"),
        ]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        return outputs + inputs


class ResNetLikeClassifier(torch.nn.Module):
    def __init__(self, num_classes: int, num_blocks: int = 6) -> None:
        super().__init__()

        cnn_layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(1, 8, (5, 7), stride=(3, 5)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, (3, 5), stride=(1, 3)),
        ] + [PreActivationResNetBlock(16, 3) for _ in range(num_blocks)]

        self.cnn_layers = torch.nn.Sequential(*cnn_layers)

        self.full = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(5120, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 128),
            torch.nn.ReLU(),
        )

        self.cls_head = torch.nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cnn_output = self.cnn_layers(inputs)
        cls_feats = self.full(cnn_output)
        return self.cls_head(cls_feats)


class ClassifierLightningModule(L.LightningModule):
    def __init__(
        self,
        target_names: list[str],
        log_metrics_every_step: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: Literal["binary_cross_entropy", "cross_entropy"] = "cross_entropy",
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        # Metrics
        self.train_metric = MultiLabelMetric(target_names)
        self.test_metric = MultiLabelMetric(target_names)
        self.val_metric = MultiLabelMetric(target_names)
        self.metrics = {
            "train": self.train_metric,
            "test": self.test_metric,
            "val": self.val_metric,
        }

        self.classifier = ResNetLikeClassifier(len(target_names))
        self.loss = (
            torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if loss == "cross_entropy"
            else torch.nn.BCEWithLogitsLoss()
        )
        self.example_input_array = torch.Tensor(32, 1, 32, 626)
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_metrics_every_step = log_metrics_every_step

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        return self._common_step("train", inputs, targets)

    def test_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        return self._common_step("test", inputs, targets)

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        return self._common_step("val", inputs, targets)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.weight_decay == 0:
            return torch.optim.Adam(params=self.classifier.parameters(), lr=self.lr)

        decay_params = []
        non_decay_params = []
        for param_name, param in self.classifier.named_parameters():
            if "bias" in param_name:
                non_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.Adam(
            params=[
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": non_decay_params},
            ],
            lr=self.lr,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)

    def _common_step(
        self,
        phase: Phase,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(inputs)

        loss = self.loss(logits, targets.to(torch.float32))
        self.log(f"{phase}/loss", loss, prog_bar=True, logger=True)

        self._update_metrics(phase, logits, targets)

        return loss.mean(dim=-1)

    def _update_metrics(
        self,
        phase: Phase,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.metrics[phase].update(logits, targets)

        if (
            phase == "train" and (self.global_step % self.log_metrics_every_step == 0)
        ) or (phase != "train" and self.trainer.is_last_batch):
            self._compute_metrics(phase)

    def _compute_metrics(self, phase: Phase):
        self.log_dict(
            {f"{phase}/{k}": v for k, v in self.metrics[phase].compute().items()},
            on_step=True,
            logger=True,
            sync_dist=True,
        )
        self.metrics[phase].reset()
