import os
import pickle
from datetime import datetime
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.data_processor import DataProcessor


class ModelTrainer:
    """Тренування регресійних моделей якості вина."""

    def __init__(self):
        self.processor = DataProcessor()
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_definitions = {
            "linear_regression": {
                "class": LinearRegression,
                "params": {},
                "description": "Лінійна регресія",
            },
            "random_forest": {
                "class": RandomForestRegressor,
                "params": {"n_estimators": 100, "random_state": 42, "max_depth": 10},
                "description": "Random Forest",
            },
            "gradient_boosting": {
                "class": GradientBoostingRegressor,
                "params": {
                    "n_estimators": 100,
                    "random_state": 42,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                },
                "description": "Gradient Boosting",
            },
        }

    def train_model(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.model_definitions:
            raise ValueError(
                f"Невідома модель: {model_name}. Доступні: {list(self.model_definitions.keys())}"
            )

        df = self.processor.load_data_from_db()
        if df.empty:
            raise ValueError("База даних порожня або недоступна")

        X, y = self.processor.preprocess_data(df, is_training=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_def = self.model_definitions[model_name]
        model = model_def["class"](**model_def["params"])
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metadata = {
            "model_name": model_name,
            "algorithm": model_def["class"].__name__,
            "training_date": datetime.now().isoformat(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "scaler": self.processor.get_scaler_params(),
            "metrics": {
                "train": {
                    "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                    "mae": float(mean_absolute_error(y_train, y_train_pred)),
                    "r2": float(r2_score(y_train, y_train_pred)),
                },
                "test": {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                    "mae": float(mean_absolute_error(y_test, y_test_pred)),
                    "r2": float(r2_score(y_test, y_test_pred)),
                },
            },
            "description": model_def["description"],
        }

        with open(os.path.join(self.models_dir, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(self.models_dir, f"{model_name}_metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        return {
            "status": "success",
            "message": f"Модель {model_name} успішно навчена",
            "metadata": metadata,
        }

    def get_available_models(self) -> list:
        return list(self.model_definitions.keys())
