from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from base_model.data import MNISTWrapper
from main import _resolve_dataset_path


def prepare_dataset_and_config(
    base_config_path: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
) -> tuple[str, dict[str, Any]]:
    """Load the dataset, validate model-ready tensors, and return updated config text."""
    base_path = Path(base_config_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(base_path)
    dataset_path = _resolve_dataset_path(OmegaConf.to_container(config.data, resolve=True))

    prepared_dataset = MNISTWrapper(data_path=dataset_path).get_dataset(
        image_size=config.training.image_size,
        mean=config.training.mean,
        std=config.training.std,
    )
    sample_tensor, sample_label = prepared_dataset["train"][0]

    config.run_mode = "local_clearml"
    config.data.local_dataset_path = str(dataset_path)
    config.data.dataset_id = ""
    config.training.num_epochs = num_epochs
    config.training.batch_size = batch_size
    config.training.resume_training = False
    config.training.resume_task_id = ""
    config.training.artifacts_dir = str(target_dir / "artifacts")

    dataset_summary = {
        "resolved_dataset_path": str(dataset_path),
        "train_size": len(prepared_dataset["train"]),
        "validation_size": len(prepared_dataset["validation"]),
        "num_classes": len(prepared_dataset["train"].classes),
        "sample_shape": list(sample_tensor.shape),
        "sample_label": int(sample_label),
    }
    return OmegaConf.to_yaml(config, resolve=True), dataset_summary
