from pathlib import Path

from clearml import Logger, PipelineDecorator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = PROJECT_ROOT / "run_config.yaml"


@PipelineDecorator.component(
    return_values=["processed_config_text", "dataset_summary"],
    cache=False,
    packages=False,
    task_type="data_processing",
)
def preprocess_data(base_config_path: str, output_dir: str, num_epochs: int, batch_size: int) -> tuple[str, dict]:
    """Load the dataset and validate its model-ready tensor conversion."""
    from pipeline_test.helpers import prepare_dataset_and_config

    processed_config_text, dataset_summary = prepare_dataset_and_config(
        base_config_path=base_config_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    Logger.current_logger().report_text(f"Prepared dataset summary: {dataset_summary}")
    return processed_config_text, dataset_summary


@PipelineDecorator.component(
    return_values=["best_model_path"],
    cache=False,
    packages=False,
    task_type="training",
)
def train_model(processed_config_text: str, output_dir: str) -> str:
    """Run the existing training code from the prepared pipeline config."""
    from main import run_training

    component_output_dir = Path(output_dir).expanduser().resolve()
    component_output_dir.mkdir(parents=True, exist_ok=True)
    processed_config_path = component_output_dir / "run_config.pipeline.yaml"
    processed_config_path.write_text(processed_config_text, encoding="utf-8")

    best_model_path = run_training(processed_config_path, initialize_task=False)
    Logger.current_logger().report_text(f"Best model saved to: {best_model_path}")
    return str(best_model_path)


@PipelineDecorator.pipeline(
    name="MNIST Pipeline Test",
    project="Custom/MNIST Pipelines",
    version="0.1",
)
def mnist_training_pipeline(
    base_config_path: str = str(BASE_CONFIG_PATH),
    output_dir: str = str(PROJECT_ROOT / "pipeline_test" / "runs"),
    num_epochs: int = 2,
    batch_size: int = 64,
) -> str:
    """Simple pipeline with explicit data preparation and model training steps."""
    processed_config_text, _ = preprocess_data(
        base_config_path=base_config_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    best_model_path = train_model(
        processed_config_text=processed_config_text,
        output_dir=output_dir,
    )
    return best_model_path


if __name__ == "__main__":
    PipelineDecorator.set_default_execution_queue("1xGPU")
    mnist_training_pipeline()
