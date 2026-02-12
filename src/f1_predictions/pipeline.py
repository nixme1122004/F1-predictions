from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "speed_mean",
    "speed_std",
    "throttle_mean",
    "brake_ratio",
    "rpm_mean",
    "gear_changes",
    "sector1_time",
    "sector2_time",
    "sector3_time",
    "track_temp",
    "air_temp",
    "humidity",
    "wind_speed",
    "fuel_load",
    "tyre_life",
]

CATEGORICAL_FEATURES = [
    "team",
    "driver",
    "compound",
    "track_status",
    "session_type",
]

TARGET_LAP_TIME = "lap_time_ms"
TARGET_RESULT_CLASS = "result_class"


@dataclass
class F1TelemetryModelBundle:
    lap_time_model: Pipeline
    result_model: Pipeline
    feature_columns: list[str]

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.lap_time_model, output_path / "lap_time_model.joblib")
        joblib.dump(self.result_model, output_path / "result_model.joblib")
        joblib.dump(self.feature_columns, output_path / "feature_columns.joblib")

    @classmethod
    def load(cls, output_dir: str | Path) -> "F1TelemetryModelBundle":
        output_path = Path(output_dir)
        return cls(
            lap_time_model=joblib.load(output_path / "lap_time_model.joblib"),
            result_model=joblib.load(output_path / "result_model.joblib"),
            feature_columns=joblib.load(output_path / "feature_columns.joblib"),
        )


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected feature columns exist and return ordered dataframe."""
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Input data is missing required columns: {missing_list}")
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def _prepare_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if TARGET_LAP_TIME not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_LAP_TIME}'")

    lap_time_target = df[TARGET_LAP_TIME].astype(float)

    if TARGET_RESULT_CLASS in df.columns:
        result_target = df[TARGET_RESULT_CLASS].astype(str)
    elif "final_position" in df.columns:
        result_target = pd.cut(
            df["final_position"],
            bins=[-np.inf, 3, 10, np.inf],
            labels=["podium", "points", "outside_points"],
        ).astype(str)
    else:
        raise ValueError(
            "Expected either 'result_class' or 'final_position' target column for classification"
        )

    return lap_time_target, result_target


def train_models(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[F1TelemetryModelBundle, dict[str, float]]:
    features = build_feature_matrix(df)
    lap_time_target, result_target = _prepare_targets(df)

    x_train, x_test, y_lap_train, y_lap_test, y_res_train, y_res_test = train_test_split(
        features,
        lap_time_target,
        result_target,
        test_size=test_size,
        random_state=random_state,
        stratify=result_target,
    )

    preprocessor = _build_preprocessor()

    lap_time_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    result_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state,
                ),
            ),
        ]
    )

    lap_time_model.fit(x_train, y_lap_train)
    result_model.fit(x_train, y_res_train)

    metrics = evaluate_models(
        lap_time_model,
        result_model,
        x_test,
        y_lap_test,
        y_res_test,
    )

    bundle = F1TelemetryModelBundle(
        lap_time_model=lap_time_model,
        result_model=result_model,
        feature_columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    )

    return bundle, metrics


def evaluate_models(
    lap_time_model: Pipeline,
    result_model: Pipeline,
    x_test: pd.DataFrame,
    y_lap_test: Iterable[float],
    y_res_test: Iterable[str],
) -> dict[str, float]:
    lap_predictions = lap_time_model.predict(x_test)
    result_predictions = result_model.predict(x_test)

    y_lap = np.asarray(list(y_lap_test), dtype=float)
    y_res = np.asarray(list(y_res_test), dtype=str)

    rmse = mean_squared_error(y_lap, lap_predictions, squared=False)
    mae = mean_absolute_error(y_lap, lap_predictions)
    acc = accuracy_score(y_res, result_predictions)
    weighted_f1 = f1_score(y_res, result_predictions, average="weighted")

    return {
        "lap_time_rmse_ms": float(rmse),
        "lap_time_mae_ms": float(mae),
        "result_accuracy": float(acc),
        "result_weighted_f1": float(weighted_f1),
    }
