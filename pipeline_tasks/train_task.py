import shutil
from pathlib import Path

from clearml import Logger, StorageManager, Task

from main import run_training


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    task = Task.init(
        project_name="Custom/MNIST Pipelines",
        task_name="MNIST_Train_Task",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={"pytorch": False},
    )

    args = {
        "processed_config_url": "",
        "output_dir": str(PROJECT_ROOT / "pipeline_tasks" / "runs" / "train"),
    }
    task.connect(args, name="Args")

    if not args["processed_config_url"]:
        raise ValueError("Set 'Args/processed_config_url' to a processed config artifact URL.")

    output_dir = Path(args["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_config_path = Path(StorageManager.get_local_copy(remote_url=args["processed_config_url"]))
    processed_config_path = output_dir / "run_config.pipeline.yaml"
    if local_config_path != processed_config_path:
        shutil.copy2(local_config_path, processed_config_path)

    best_model_path = run_training(processed_config_path, initialize_task=False)
    Logger.current_logger().report_text(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
