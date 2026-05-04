from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from base_model.data import MNISTWrapper
from base_model.inference import (
    DEFAULT_CONFIG_PATH,
    _load_config,
    _resolve_checkpoint_path,
    _resolve_dataset_path,
)
from base_model.model import BWResNet18Wrapper


def _build_inference_transform(image_size: int, mean: float, std: float) -> transforms.Compose:
    """Mirror the training-time image preprocessing for online inference."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


class OnlinePredictor:
    """Load the trained classifier once and serve single-image predictions."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path.resolve()
        self.config = _load_config(self.config_path)
        self.config_dir = self.config_path.parent
        self.training_config = self.config["training"]
        self.dataset_path = _resolve_dataset_path(self.config["data"], base_dir=self.config_dir)
        self.checkpoint_path = _resolve_checkpoint_path(self.config, base_dir=self.config_dir)
        self.class_names = self._load_class_names()
        self.transform = _build_inference_transform(
            image_size=int(self.training_config["image_size"]),
            mean=float(self.training_config["mean"]),
            std=float(self.training_config["std"]),
        )

        self.model_wrapper = BWResNet18Wrapper(
            num_classes=len(self.class_names),
            pretrained=bool(self.training_config["pretrained"]),
            hid_lay_size=int(self.training_config["hid_lay_size"]),
            dropout=float(self.training_config["dropout"]),
            artifacts_dir=self.training_config["artifacts_dir"],
            weights_path=str(self.checkpoint_path),
        )
        self.model_wrapper.model.eval()

    def _load_class_names(self) -> list[str]:
        """Read class labels from the training dataset metadata."""
        dataset = MNISTWrapper(data_path=self.dataset_path).get_dataset(
            image_size=int(self.training_config["image_size"]),
            mean=float(self.training_config["mean"]),
            std=float(self.training_config["std"]),
        )
        for split_name in ("train", "validation"):
            split_dataset = dataset.get(split_name)
            if split_dataset is not None and hasattr(split_dataset, "classes"):
                return [str(class_name) for class_name in split_dataset.classes]
        raise ValueError("Could not infer class names from the configured dataset.")

    @staticmethod
    def _decode_base64_image(image_base64: str) -> Image.Image:
        """Decode a base64 image string into a PIL image."""
        encoded_image = image_base64
        if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
            encoded_image = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded_image)
        return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def _load_request_image(body: bytes | dict[str, Any]) -> Image.Image:
        """Support raw bytes, base64 JSON, or a local image path for testing."""
        if isinstance(body, bytes):
            return Image.open(io.BytesIO(body))

        image_base64 = body.get("image_base64")
        if image_base64:
            return OnlinePredictor._decode_base64_image(str(image_base64))

        image_path = body.get("image_path")
        if image_path:
            return Image.open(Path(str(image_path)).expanduser().resolve())

        raise ValueError("Request must include raw image bytes, 'image_base64', or 'image_path'.")

    def predict_image(self, image: Image.Image, *, top_k: int = 3) -> dict[str, Any]:
        """Return ranked class probabilities for a single image."""
        input_tensor = self.transform(image).unsqueeze(0).to(self.model_wrapper.device)

        with torch.no_grad():
            logits = self.model_wrapper.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

        requested_top_k = max(1, int(top_k))
        actual_top_k = min(requested_top_k, len(self.class_names))
        top_probabilities, top_indices = torch.topk(probabilities, k=actual_top_k)

        predictions = []
        for confidence, class_index in zip(top_probabilities.tolist(), top_indices.tolist()):
            predictions.append(
                {
                    "class_index": int(class_index),
                    "label": self.class_names[int(class_index)],
                    "confidence": float(confidence),
                }
            )

        return {
            "predicted_class_index": predictions[0]["class_index"],
            "predicted_label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def predict_request(self, body: bytes | dict[str, Any]) -> dict[str, Any]:
        """Parse an external request and return a JSON-serializable prediction."""
        request_payload = body if isinstance(body, dict) else {}
        top_k = int(request_payload.get("top_k", 3))
        image = self._load_request_image(body)
        return self.predict_image(image, top_k=top_k)
