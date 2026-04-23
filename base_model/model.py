import re
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from clearml import Task
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .resnet18 import BWResNet18


class BWResNet18Wrapper:
    """Train and run a single-channel ResNet18 classifier."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        hid_lay_size: int = 100,
        dropout: float = 0.1,
        artifacts_dir: str | Path = "artifacts",
        weights_path: str | None = None,
    ) -> None:
        self.model = BWResNet18(
            n_classes=num_classes,
            pretrained=pretrained,
            hid_lay_size=hid_lay_size,
            dropout=dropout,
        )
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to_device(self.device)

        if weights_path:
            checkpoint_path = self.artifacts_dir / weights_path
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self._load_model_state(checkpoint)

    @staticmethod
    def _extract_epoch_from_name(name: str) -> int | None:
        """Parse checkpoint names like model_0 or model_0.pth."""
        match = re.search(r"model_(\d+)(?:\.pth)?$", name)
        if match is None:
            return None
        return int(match.group(1))

    def _load_model_state(self, checkpoint: Any) -> None:
        """Load model weights from either a plain state dict or a training checkpoint."""
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            return
        self.model.load_state_dict(checkpoint)

    def _get_local_checkpoint(self) -> tuple[Path | None, int]:
        """Return the most recent local epoch checkpoint path."""
        latest_path: Path | None = None
        latest_epoch = -1

        for checkpoint_path in self.artifacts_dir.glob("model_*.pth"):
            epoch = self._extract_epoch_from_name(checkpoint_path.name)
            if epoch is None or epoch <= latest_epoch:
                continue
            latest_epoch = epoch
            latest_path = checkpoint_path

        return latest_path, latest_epoch

    def _get_task_checkpoint(self, task: Task | None) -> tuple[Path | None, int]:
        """Download the most recent epoch checkpoint artifact from ClearML."""
        if task is None:
            return None, -1

        task.reload()
        latest_artifact = None
        latest_epoch = -1

        for artifact_name, artifact in task.artifacts.items():
            epoch = self._extract_epoch_from_name(artifact_name)
            if epoch is None or epoch <= latest_epoch:
                continue
            latest_epoch = epoch
            latest_artifact = artifact

        if latest_artifact is None:
            return None, -1

        downloaded_checkpoint = Path(latest_artifact.get_local_copy(raise_on_error=True, force_download=True))
        local_checkpoint = self.artifacts_dir / f"model_{latest_epoch}.pth"
        if downloaded_checkpoint != local_checkpoint:
            shutil.copy2(downloaded_checkpoint, local_checkpoint)

        return local_checkpoint, latest_epoch

    def _resume_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingLR,
        resume_task_id: str | None = None,
    ) -> tuple[int, float]:
        """Resume from the newest available checkpoint and return the next epoch and best accuracy."""
        resume_task = Task.current_task()
        if resume_task_id:
            current_task = Task.current_task()
            if current_task is None or current_task.id != resume_task_id:
                resume_task = Task.get_task(task_id=resume_task_id)

        local_checkpoint_path, local_epoch = self._get_local_checkpoint()
        task_checkpoint_path, task_epoch = self._get_task_checkpoint(resume_task)

        checkpoint_path = local_checkpoint_path
        completed_epoch = local_epoch
        if task_epoch > completed_epoch:
            checkpoint_path = task_checkpoint_path
            completed_epoch = task_epoch

        if checkpoint_path is None:
            return 0, -1.0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._load_model_state(checkpoint)

        best_accuracy = -1.0
        if isinstance(checkpoint, dict):
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
            completed_epoch = int(checkpoint.get("epoch", completed_epoch))

        return completed_epoch + 1, best_accuracy

    def train_function(
        self,
        dataset: Dict[str, Dataset],
        lr: float = 0.001,
        momentum: float = 0.9,
        num_epochs: int = 5,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 2,
        print_metrics: bool = False,
        resume_training: bool = False,
        resume_task_id: str | None = None,
    ) -> Path:
        """Run a standard supervised training loop and return the best checkpoint path."""
        task = Task.current_task()
        logger = task.get_logger() if task is not None else None

        train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        validation_loader = DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, len(train_loader) * num_epochs),
        )

        start_epoch = 0
        best_accuracy = -1.0
        best_model_path = self.artifacts_dir / "best_model.pth"
        if resume_training:
            start_epoch, best_accuracy = self._resume_training_state(
                optimizer=optimizer,
                scheduler=scheduler,
                resume_task_id=resume_task_id,
            )
            if start_epoch >= num_epochs:
                if not best_model_path.exists():
                    latest_checkpoint_path, _ = self._get_local_checkpoint()
                    if latest_checkpoint_path is not None:
                        shutil.copy2(latest_checkpoint_path, best_model_path)
                return best_model_path

        for epoch_num in range(start_epoch, num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

            train_loss = running_loss / max(1, total)
            train_accuracy = 100.0 * correct / max(1, total)
            validation_loss, validation_accuracy = self.validation(validation_loader)
            current_lr = optimizer.param_groups[0]["lr"]

            if logger is not None:
                logger.report_scalar("loss", "train", iteration=epoch_num, value=train_loss)
                logger.report_scalar(
                    "accuracy",
                    "train",
                    iteration=epoch_num,
                    value=train_accuracy,
                )
                logger.report_scalar(
                    "loss",
                    "validation",
                    iteration=epoch_num,
                    value=validation_loss,
                )
                logger.report_scalar(
                    "accuracy",
                    "validation",
                    iteration=epoch_num,
                    value=validation_accuracy,
                )
                logger.report_scalar("lr", "train", iteration=epoch_num, value=current_lr)
            elif print_metrics:
                print(
                    f"epoch={epoch_num} "
                    f"lr={current_lr:.6f} "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_accuracy:.2f} "
                    f"val_loss={validation_loss:.4f} "
                    f"val_acc={validation_accuracy:.2f}"
                )

            if validation_accuracy >= best_accuracy:
                best_accuracy = validation_accuracy
                torch.save(
                    {
                        "epoch": epoch_num,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_accuracy": best_accuracy,
                    },
                    best_model_path,
                )

            checkpoint_path = self.artifacts_dir / f"model_{epoch_num}.pth"
            torch.save(
                {
                    "epoch": epoch_num,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                },
                checkpoint_path,
            )
            if task is not None:
                task.upload_artifact(f"model_{epoch_num}", artifact_object=str(checkpoint_path))

        return best_model_path

    def validation(self, validation_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on the validation split."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        validation_loss = running_loss / max(1, total)
        validation_accuracy = 100.0 * correct / max(1, total)
        return validation_loss, validation_accuracy

    def predict_function(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of tensors shaped like the training data."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_batch.to(self.device))
        return outputs.argmax(dim=1)

    def to_device(self, device: str) -> None:
        """Move the model to the requested device."""
        self.device = device
        self.model.to(device)
