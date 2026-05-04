from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from clearml import Dataset, Logger, Model, Task
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset as TorchDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_model.data import MNISTWrapper
from base_model.model import BWResNet18Wrapper


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "run_config.yaml"


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load the runtime configuration from disk."""
    config = OmegaConf.load(config_path)
    return dict(OmegaConf.to_container(config, resolve=True))


def _get_inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return inference config or an empty default block."""
    inference_config = config.get("inference", {})
    if not isinstance(inference_config, dict):
        raise ValueError("'inference' must be a mapping in the config file.")
    return inference_config


def _resolve_path(path_value: str, *, base_dir: Path) -> Path:
    """Resolve a local path relative to the config directory when needed."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _download_dataset(dataset_id: str) -> Path:
    """Download a ClearML dataset and return the local dataset directory."""
    dataset = Dataset.get(dataset_id=dataset_id)
    return Path(dataset.get_local_copy())


def _resolve_dataset_path(data_config: dict[str, Any], *, base_dir: Path) -> Path:
    """Resolve the dataset path from either local storage or a ClearML dataset id."""
    local_dataset_path = str(data_config.get("local_dataset_path", "")).strip()
    if local_dataset_path:
        return _resolve_path(local_dataset_path, base_dir=base_dir)

    dataset_id = str(data_config.get("dataset_id", "")).strip()
    if dataset_id:
        return _download_dataset(dataset_id)

    raise ValueError("Set either 'data.local_dataset_path' or 'data.dataset_id' in the config file.")


def _resolve_clearml_checkpoint(model_id: str) -> Path:
    """Download a checkpoint from a ClearML model id."""
    model = Model(model_id=model_id)
    return Path(model.get_local_copy(raise_on_error=True, force_download=True))


def _extract_epoch_from_name(path: Path) -> int | None:
    """Parse checkpoint names like model_0.pth."""
    match = re.search(r"model_(\d+)\.pth$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _find_latest_checkpoint(artifacts_dir: Path) -> Path | None:
    """Fallback to the latest epoch checkpoint when best_model is unavailable."""
    latest_checkpoint: Path | None = None
    latest_epoch = -1

    for checkpoint_path in artifacts_dir.glob("model_*.pth"):
        epoch = _extract_epoch_from_name(checkpoint_path)
        if epoch is None or epoch <= latest_epoch:
            continue
        latest_epoch = epoch
        latest_checkpoint = checkpoint_path

    return latest_checkpoint


def _resolve_checkpoint_path(config: dict[str, Any], *, base_dir: Path) -> Path:
    """Resolve a model checkpoint from config or the training artifacts directory."""
    model_config = config["model"]
    training_config = config["training"]

    local_weights_path = str(model_config.get("local_weights_path", "")).strip()
    if local_weights_path:
        checkpoint_path = _resolve_path(local_weights_path, base_dir=base_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Configured weights path does not exist: {checkpoint_path}")
        return checkpoint_path

    clearml_model_id = str(model_config.get("clearml_model_id", "")).strip()
    if clearml_model_id:
        return _resolve_clearml_checkpoint(clearml_model_id)

    artifacts_dir = _resolve_path(str(training_config["artifacts_dir"]), base_dir=base_dir)
    best_model_path = artifacts_dir / "best_model.pth"
    if best_model_path.exists():
        return best_model_path

    latest_checkpoint = _find_latest_checkpoint(artifacts_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint

    raise FileNotFoundError(
        "Could not resolve a checkpoint. Set 'model.local_weights_path', "
        "'model.clearml_model_id', or place a checkpoint in the artifacts directory."
    )


def _init_task_if_needed(
    config: dict[str, Any],
    *,
    initialize_task: bool,
    split: str,
) -> Task | None:
    """Initialize a ClearML task only when inference logging is enabled."""
    if not initialize_task:
        return Task.current_task()

    inference_config = _get_inference_config(config)
    run_mode = str(inference_config.get("run_mode", "")).strip() or config["run_mode"]
    if run_mode == "local_no_clearml":
        return None

    clearml_config = config["clearml"]
    task_name_prefix = str(inference_config.get("task_name", "")).strip() or clearml_config["task_name"]
    task = Task.init(
        project_name=clearml_config["project_name"],
        task_name=f"{task_name_prefix}_{split}_inference",
        auto_connect_frameworks={"pytorch": False},
    )
    task.connect(config, name="run_config")

    docker_image = str(clearml_config.get("docker_image", "")).strip()
    if docker_image:
        task.set_base_docker(
            docker_image=docker_image,
            docker_arguments=clearml_config.get("docker_arguments", ""),
        )

    if run_mode == "remote_clearml":
        queue_name = str(clearml_config.get("queue_name", "")).strip()
        if not queue_name:
            raise ValueError("Set 'clearml.queue_name' for 'remote_clearml' mode.")
        task.execute_remotely(queue_name=queue_name, clone=False)

    return task


def _get_split_dataset(
    dataset_path: Path,
    *,
    image_size: int,
    mean: float,
    std: float,
    split: str,
) -> TorchDataset:
    """Build datasets with the training transforms and return the requested split."""
    datasets = MNISTWrapper(data_path=dataset_path).get_dataset(
        image_size=image_size,
        mean=mean,
        std=std,
    )
    if split not in datasets:
        available_splits = ", ".join(sorted(datasets))
        raise ValueError(f"Unsupported split '{split}'. Available splits: {available_splits}")
    return datasets[split]


def run_inference(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    split: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    initialize_task: bool = True,
) -> dict[str, float | int | str]:
    """Evaluate a saved checkpoint on a dataset split and return the metrics."""
    config = _load_config(config_path)
    config_dir = config_path.resolve().parent
    training_config = config["training"]
    inference_config = _get_inference_config(config)
    eval_split = split or str(inference_config.get("split", "validation"))

    task = _init_task_if_needed(config, initialize_task=initialize_task, split=eval_split)
    dataset_path = _resolve_dataset_path(config["data"], base_dir=config_dir)
    checkpoint_path = _resolve_checkpoint_path(config, base_dir=config_dir)
    dataset_split = _get_split_dataset(
        dataset_path,
        image_size=training_config["image_size"],
        mean=training_config["mean"],
        std=training_config["std"],
        split=eval_split,
    )

    eval_batch_size = batch_size or int(inference_config.get("batch_size", training_config["batch_size"]))
    eval_num_workers = (
        num_workers if num_workers is not None else int(inference_config.get("num_workers", training_config["num_workers"]))
    )
    dataloader = DataLoader(
        dataset_split,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
    )

    evaluator = BWResNet18Wrapper(
        num_classes=len(dataset_split.classes),
        pretrained=training_config["pretrained"],
        hid_lay_size=training_config["hid_lay_size"],
        dropout=training_config["dropout"],
        artifacts_dir=training_config["artifacts_dir"],
        weights_path=str(checkpoint_path),
    )
    loss, accuracy = evaluator.validation(dataloader)

    metrics: dict[str, float | int | str] = {
        "split": eval_split,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "num_samples": len(dataset_split),
        "batch_size": eval_batch_size,
        "loss": loss,
        "accuracy": accuracy,
    }

    if task is not None:
        logger = Logger.current_logger()
        logger.report_scalar("loss", eval_split, iteration=0, value=loss)
        logger.report_scalar("accuracy", eval_split, iteration=0, value=accuracy)
        task.connect(
            {
                "split": eval_split,
                "checkpoint_path": str(checkpoint_path),
                "dataset_path": str(dataset_path),
                "num_samples": len(dataset_split),
                "classes": list(dataset_split.classes),
            },
            name="inference_metadata",
        )

    print(f"split={eval_split}")
    print(f"checkpoint={checkpoint_path}")
    print(f"dataset={dataset_path}")
    print(f"samples={len(dataset_split)}")
    print(f"loss={loss:.4f}")
    print(f"accuracy={accuracy:.2f}")

    return metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a dataset split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=("train", "validation"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for the evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for the dataloader worker count.",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Skip ClearML task initialization even if the config enables it.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_inference(
        config_path=args.config.resolve(),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        initialize_task=not args.no_clearml,
    )


if __name__ == "__main__":
    main()
