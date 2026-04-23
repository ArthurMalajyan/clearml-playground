from pathlib import Path
from typing import Dict

import torch
from clearml import Dataset as ClearMLDataset
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
        weights_path: str = "best_model.pth",
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

        checkpoint_path = self.artifacts_dir / weights_path
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

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

        best_accuracy = -1.0
        best_model_path = self.artifacts_dir / "best_model.pth"

        for epoch_num in range(num_epochs):
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

            checkpoint_path = self.artifacts_dir / f"model_{epoch_num}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)
            if task is not None:
                task.upload_artifact(f"model_{epoch_num}", artifact_object=str(checkpoint_path))

            if validation_accuracy >= best_accuracy:
                best_accuracy = validation_accuracy
                torch.save(self.model.state_dict(), best_model_path)

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
