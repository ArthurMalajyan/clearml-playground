from pathlib import Path
from typing import Dict, Iterable

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.mnist import MNIST


class CachedMNIST(MNIST):
    """MNIST loader that reads directly from an arbitrary dataset directory."""

    def __init__(
        self,
        data_dir: str | Path,
        train: bool,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        super().__init__(root=str(self.data_dir), train=train, download=False, transform=transform)

    @property
    def raw_folder(self) -> str:
        return str(self.data_dir / "raw")

    @property
    def processed_folder(self) -> str:
        return str(self.data_dir / "processed")


class MNISTWrapper:
    """Load an image classification dataset from a ClearML dataset copy."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def _resolve_torchvision_mnist_root(self) -> Path | None:
        """Return the directory that contains MNIST raw/processed files."""
        if (self.data_path / "raw").is_dir() or (self.data_path / "processed").is_dir():
            return self.data_path

        if (self.data_path / "MNIST" / "raw").is_dir() or (self.data_path / "MNIST" / "processed").is_dir():
            return self.data_path / "MNIST"

        return None

    def _find_split_dir(self, split_names: Iterable[str]) -> Path:
        """Resolve a split directory either at the root or one level below it."""
        for split_name in split_names:
            direct_path = self.data_path / split_name
            if direct_path.is_dir():
                return direct_path

        for child_dir in self.data_path.iterdir():
            if not child_dir.is_dir():
                continue
            for split_name in split_names:
                nested_path = child_dir / split_name
                if nested_path.is_dir():
                    return nested_path

        split_hint = ", ".join(split_names)
        raise FileNotFoundError(
            f"Could not find a dataset split named one of [{split_hint}] under {self.data_path}"
        )

    def get_dataset(
        self,
        image_size: int = 224,
        mean: float = 0.5,
        std: float = 0.5,
    ) -> Dict[str, Dataset]:
        """Return train and validation datasets ready for a 1-channel ResNet."""
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        )

        mnist_root = self._resolve_torchvision_mnist_root()

        if mnist_root is not None:
            return {
                "train": CachedMNIST(
                    data_dir=mnist_root,
                    train=True,
                    transform=transform,
                ),
                "validation": CachedMNIST(
                    data_dir=mnist_root,
                    train=False,
                    transform=transform,
                ),
            }

        train_dir = self._find_split_dir(("train",))
        validation_dir = self._find_split_dir(("validation", "val", "test"))

        return {
            "train": ImageFolder(root=train_dir, transform=transform),
            "validation": ImageFolder(root=validation_dir, transform=transform),
        }
