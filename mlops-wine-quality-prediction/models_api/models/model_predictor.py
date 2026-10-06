import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd

from utils.data_processor import DataProcessor


class ModelPredictor:
    """Прогноз якості вина навченими моделями."""

    def __init__(self):
        self.processor = DataProcessor()
        self.models_dir = "models"

    def get_trained_models(self) -> list:
        if not os.path.isdir(self.models_dir):
            return []
        trained = []
        for fname in os.listdir(self.models_dir):
            if fname.endswith(".pkl") and not fname.endswith("_metadata.pkl"):
                trained.append(fname[:-4])
        return sorted(trained)

    def load_model(self, model_name: str):
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель {model_name} не знайдена. Спочатку потрібно її навчити."
            )
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def load_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.pkl")
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, "rb") as f:
            return pickle.load(f)

    def predict(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        is_valid, error_message = self.processor.validate_input_data(input_data)
        if not is_valid:
            raise ValueError(error_message)

        metadata = self.load_metadata(model_name)
        if metadata is None or "scaler" not in metadata:
            raise FileNotFoundError(
                f"Метадані або параметри масштабування для {model_name} не знайдені. "
                "Навчіть модель ще раз."
            )
        self.processor.set_scaler_params(metadata["scaler"])

        model = self.load_model(model_name)
        X, _ = self.processor.preprocess_data(
            pd.DataFrame([input_data]),
            is_training=False,
        )
        prediction = model.predict(X)[0]
        prediction_rounded = max(0, min(10, round(prediction)))

        return {
            "model_name": model_name,
            "prediction": float(prediction),
            "prediction_rounded": int(prediction_rounded),
            "input_data": input_data,
        }

    def get_metrics(self, model_name: str) -> Dict[str, Any]:
        metadata = self.load_metadata(model_name)
        if metadata is None:
            raise FileNotFoundError(
                f"Метадані для моделі {model_name} не знайдені. Модель не була навчена."
            )
        return {
            "model_name": model_name,
            "algorithm": metadata["algorithm"],
            "training_date": metadata["training_date"],
            "description": metadata["description"],
            "metrics": metadata["metrics"],
            "training_info": {
                "train_samples": metadata["train_samples"],
                "test_samples": metadata["test_samples"],
            },
        }
