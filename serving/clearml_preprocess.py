from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_model.serving import OnlinePredictor


class Preprocess:
    """ClearML Serving adapter for a custom PyTorch image classifier."""

    def __init__(self) -> None:
        self.model_endpoint = None
        self.predictor: OnlinePredictor | None = None

    @staticmethod
    def _resolve_config_path() -> Path:
        """Resolve the runtime config path from env or fall back to run_config.yaml."""
        config_override = os.environ.get("CLEARML_SERVING_CONFIG_PATH", "").strip()
        if config_override:
            return Path(config_override).expanduser().resolve()
        return (Path(__file__).resolve().parent.parent / "run_config.yaml").resolve()

    def preprocess(
        self,
        body: bytes | dict[str, Any],
        state: dict[str, Any],
        collect_custom_statistics_fn: Optional[Callable[[dict[str, Any]], None]],
    ) -> bytes | dict[str, Any]:
        """Preserve request metadata and pass the payload through unchanged."""
        if isinstance(body, dict):
            state["requested_top_k"] = int(body.get("top_k", 3))
            state["request_keys"] = sorted(body.keys())
        else:
            state["requested_top_k"] = 3
            state["request_keys"] = ["<raw-bytes>"]

        if collect_custom_statistics_fn is not None:
            collect_custom_statistics_fn({"request_type": "dict" if isinstance(body, dict) else "bytes"})

        return body

    def process(
        self,
        data: bytes | dict[str, Any],
        state: dict[str, Any],
        collect_custom_statistics_fn: Optional[Callable[[dict[str, Any]], None]],
    ) -> dict[str, Any]:
        """Load the model once and run a single prediction request."""
        if self.predictor is None:
            self.predictor = OnlinePredictor(config_path=self._resolve_config_path())

        result = self.predictor.predict_request(data)
        result["requested_top_k"] = state["requested_top_k"]

        if collect_custom_statistics_fn is not None:
            collect_custom_statistics_fn(
                {
                    "predicted_class_index": result["predicted_class_index"],
                    "confidence": result["confidence"],
                }
            )

        return result

    def postprocess(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        collect_custom_statistics_fn: Optional[Callable[[dict[str, Any]], None]],
    ) -> dict[str, Any]:
        """Return the prediction payload as the HTTP response."""
        return {
            "prediction": data,
            "request_keys": state.get("request_keys", []),
        }
