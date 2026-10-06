import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine


class DataProcessor:
    """Завантаження, валідація та масштабування ознак якості вина."""

    FEATURE_RANGES = {
        "fixed_acidity": (4.0, 16.0),
        "volatile_acidity": (0.1, 2.0),
        "citric_acid": (0.0, 1.0),
        "residual_sugar": (0.5, 15.0),
        "chlorides": (0.01, 0.6),
        "free_sulfur_dioxide": (1.0, 72.0),
        "total_sulfur_dioxide": (6.0, 289.0),
        "density": (0.99, 1.004),
        "pH": (2.7, 4.0),
        "sulphate": (0.2, 2.0),
        "alcohol": (8.0, 15.0),
    }

    def __init__(self):
        self.feature_names = list(self.FEATURE_RANGES.keys())
        self.target_name = "quality"
        self.min_values: Optional[pd.Series] = None
        self.max_values: Optional[pd.Series] = None

    def get_db_engine(self):
        user = os.getenv("DB_USER", "mlops_user")
        password = os.getenv("DB_PASSWORD", "mlops_password")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "mlops_db")
        return create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )

    def load_data_from_db(self) -> pd.DataFrame:
        engine = self.get_db_engine()
        columns = ", ".join(
            "ph" if col == "pH" else col for col in self.feature_names
        )
        query = f"SELECT {columns}, {self.target_name} FROM wine_data"
        df = pd.read_sql_query(query, engine)
        df = df.rename(columns={"ph": "pH"})
        df.columns = [col if col == "pH" else col.lower() for col in df.columns]

        required = self.feature_names + [self.target_name]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Відсутні колонки: {missing}. Наявні: {list(df.columns)}"
            )
        if df.empty:
            raise ValueError("База даних повернула порожній результат")

        df = df.dropna(subset=required)
        if df.empty:
            raise ValueError("Після видалення порожніх значень дані відсутні")
        return df[required]

    def get_scaler_params(self) -> Dict[str, Dict[str, float]]:
        if self.min_values is None or self.max_values is None:
            raise ValueError("Параметри масштабування ще не обчислені")
        return {
            "min": self.min_values.to_dict(),
            "max": self.max_values.to_dict(),
        }

    def set_scaler_params(self, scaler: Dict[str, Dict[str, float]]) -> None:
        self.min_values = pd.Series(scaler["min"])
        self.max_values = pd.Series(scaler["max"])

    def _normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        ranges = self.max_values - self.min_values
        ranges = ranges.replace(0, 1.0)
        return (X - self.min_values) / ranges

    def preprocess_data(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> Tuple[Any, Optional[Any]]:
        df_processed = df.copy().drop_duplicates()

        if is_training:
            for col in self.feature_names:
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                if std == 0 or pd.isna(std):
                    continue
                df_processed = df_processed[
                    (df_processed[col] >= mean - 3 * std)
                    & (df_processed[col] <= mean + 3 * std)
                ]
            X = df_processed[self.feature_names]
            self.min_values = X.min()
            self.max_values = X.max()
            X_normalized = self._normalize(X)
            return X_normalized.values, df_processed[self.target_name].values

        if self.min_values is None or self.max_values is None:
            raise ValueError("Немає параметрів масштабування для прогнозу")
        X = df_processed[self.feature_names]
        return self._normalize(X).values, None

    def validate_input_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        missing_fields = [field for field in self.feature_names if field not in data]
        if missing_fields:
            return False, f"Відсутні поля: {', '.join(missing_fields)}"

        for field, (min_val, max_val) in self.FEATURE_RANGES.items():
            try:
                value = float(data[field])
            except (TypeError, ValueError):
                return False, f"Поле {field} має некоректний тип"
            if not (min_val <= value <= max_val):
                return False, (
                    f"Поле {field} має значення {value}, "
                    f"яке виходить за межі [{min_val}, {max_val}]"
                )
        return True, ""
