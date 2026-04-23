from pathlib import Path
from typing import Any

from clearml import Dataset, Task
from omegaconf import OmegaConf

from base_model.data import MNISTWrapper
from base_model.model import BWResNet18Wrapper


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "run_config.yaml"


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load the training configuration from disk."""
    config = OmegaConf.load(config_path)
    return dict(OmegaConf.to_container(config, resolve=True))


def _download_dataset(dataset_id: str) -> Path:
    """Download a ClearML dataset and return the local dataset directory."""
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = Path(dataset.get_local_copy())
    return dataset_path


def _resolve_dataset_path(data_config: dict[str, Any]) -> Path:
    """Resolve the dataset path from either a local path or a ClearML dataset id."""
    local_dataset_path = str(data_config.get("local_dataset_path", "")).strip()
    if local_dataset_path:
        return Path(local_dataset_path).expanduser().resolve()

    dataset_id = str(data_config.get("dataset_id", "")).strip()
    if dataset_id:
        return _download_dataset(dataset_id)

    raise ValueError("Set either 'data.local_dataset_path' or 'data.dataset_id' in run_config.yaml.")


def _init_task_if_needed(config: dict[str, Any]) -> Task | None:
    """Initialize a ClearML task only for modes that require logging."""
    run_mode = config["run_mode"]
    clearml_config = config["clearml"]

    if run_mode == "local_no_clearml":
        return None

    task = Task.init(
        project_name=clearml_config["project_name"],
        task_name=clearml_config["task_name"],
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


def main() -> None:
    config = _load_config(CONFIG_PATH)
    task = _init_task_if_needed(config)

    dataset_path = _resolve_dataset_path(config["data"])
    training_config = config["training"]

    dataset = MNISTWrapper(data_path=dataset_path).get_dataset(
        image_size=training_config["image_size"],
        mean=training_config["mean"],
        std=training_config["std"],
    )

    if task is not None:
        task.connect({"classes": dataset["train"].classes}, name="dataset_metadata")

    trainer = BWResNet18Wrapper(
        num_classes=len(dataset["train"].classes),
        pretrained=training_config["pretrained"],
        hid_lay_size=training_config["hid_lay_size"],
        dropout=training_config["dropout"],
        artifacts_dir=training_config["artifacts_dir"],
    )
    best_model_path = trainer.train_function(
        dataset=dataset,
        lr=training_config["lr"],
        momentum=training_config["momentum"],
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        print_metrics=config["run_mode"] == "local_no_clearml",
    )

    if task is not None:
        task.upload_artifact("best_model", artifact_object=str(best_model_path))


if __name__ == "__main__":
    main()
