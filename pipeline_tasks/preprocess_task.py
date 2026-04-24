from pathlib import Path

from clearml import Logger, Task

from pipeline_test.helpers import prepare_dataset_and_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "run_config.yaml"


def main() -> None:
    task = Task.init(
        project_name="Custom/MNIST Pipelines",
        task_name="MNIST_Preprocess_Task",
        task_type=Task.TaskTypes.data_processing,
    )

    args = {
        "base_config_path": str(DEFAULT_CONFIG_PATH),
        "output_dir": str(PROJECT_ROOT / "pipeline_tasks" / "runs" / "preprocess"),
        "dataset_id": "df6e95d13caf49dabffaa66070c3ce59",
        "num_epochs": 2,
        "batch_size": 64,
    }
    task.connect(args, name="Args")

    output_dir = Path(args["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_config_text, dataset_summary = prepare_dataset_and_config(
        base_config_path=args["base_config_path"],
        output_dir=str(output_dir),
        dataset_id=args["dataset_id"],
        num_epochs=args["num_epochs"],
        batch_size=args["batch_size"],
    )

    processed_config_path = output_dir / "run_config.pipeline.yaml"
    processed_config_path.write_text(processed_config_text, encoding="utf-8")

    logger = Logger.current_logger()
    logger.report_text(f"Prepared dataset summary: {dataset_summary}")
    task.upload_artifact("processed_config", artifact_object=str(processed_config_path))
    task.upload_artifact("dataset_summary", artifact_object=dataset_summary)


if __name__ == "__main__":
    main()
