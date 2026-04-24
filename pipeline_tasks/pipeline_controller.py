from pathlib import Path

from clearml import Task
from clearml.automation.controller import PipelineController


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    task = Task.init(
        project_name="Custom/MNIST Pipelines",
        task_name="MNIST_Task_Based_Pipeline",
        task_type=Task.TaskTypes.controller,
    )

    args = {
        "base_config_path": str(PROJECT_ROOT / "run_config.yaml"),
        "pipeline_output_dir": str(PROJECT_ROOT / "pipeline_tasks" / "runs"),
        "dataset_id": "df6e95d13caf49dabffaa66070c3ce59",
        "num_epochs": 2,
        "batch_size": 64,
        "step_queue": "1xGPU",
        "controller_queue": "1xGPU",
    }
    task.connect(args, name="Args")

    pipeline = PipelineController(
        name="MNIST Task Pipeline",
        project="Custom/MNIST Pipelines",
        version="0.1",
        add_pipeline_tags=True,
    )
    pipeline.set_default_execution_queue(args["step_queue"])

    pipeline.add_step(
        name="preprocess",
        base_task_project="Custom/MNIST Pipelines",
        base_task_name="MNIST_Preprocess_Task",
        parameter_override={
            "Args/base_config_path": args["base_config_path"],
            "Args/output_dir": str(Path(args["pipeline_output_dir"]) / "preprocess"),
            "Args/dataset_id": args["dataset_id"],
            "Args/num_epochs": args["num_epochs"],
            "Args/batch_size": args["batch_size"],
        },
        execution_queue=args["step_queue"],
        monitor_artifacts=["processed_config", "dataset_summary"],
    )

    pipeline.add_step(
        name="train",
        base_task_project="Custom/MNIST Pipelines",
        base_task_name="MNIST_Train_Task",
        parents=["preprocess"],
        parameter_override={
            "Args/processed_config_url": "${preprocess.artifacts.processed_config.url}",
            "Args/output_dir": str(Path(args["pipeline_output_dir"]) / "train"),
        },
        execution_queue=args["step_queue"],
        monitor_models=["best_model", "model_*"],
    )

    pipeline.start(queue=args["controller_queue"])


if __name__ == "__main__":
    main()
