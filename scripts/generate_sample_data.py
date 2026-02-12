#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main(rows: int = 1500, output_path: str = "data/sample_telemetry_features.csv") -> None:
    rng = np.random.default_rng(42)

    teams = ["Red Bull", "Ferrari", "Mercedes", "McLaren", "Aston Martin"]
    drivers = [
        "VER",
        "PER",
        "LEC",
        "SAI",
        "HAM",
        "RUS",
        "NOR",
        "PIA",
        "ALO",
        "STR",
    ]
    compounds = ["SOFT", "MEDIUM", "HARD"]
    track_status = ["GREEN", "YELLOW", "VSC", "SC"]
    session_type = ["RACE", "SPRINT"]

    df = pd.DataFrame(
        {
            "speed_mean": rng.normal(238, 12, rows),
            "speed_std": rng.normal(14, 4, rows),
            "throttle_mean": rng.uniform(0.52, 0.93, rows),
            "brake_ratio": rng.uniform(0.06, 0.25, rows),
            "rpm_mean": rng.normal(11500, 550, rows),
            "gear_changes": rng.integers(35, 70, rows),
            "sector1_time": rng.normal(28500, 1200, rows),
            "sector2_time": rng.normal(31500, 1300, rows),
            "sector3_time": rng.normal(33500, 1500, rows),
            "track_temp": rng.normal(37, 7, rows),
            "air_temp": rng.normal(26, 5, rows),
            "humidity": rng.uniform(18, 72, rows),
            "wind_speed": rng.uniform(1, 10, rows),
            "fuel_load": rng.uniform(4, 105, rows),
            "tyre_life": rng.integers(1, 32, rows),
            "team": rng.choice(teams, rows),
            "driver": rng.choice(drivers, rows),
            "compound": rng.choice(compounds, rows, p=[0.38, 0.42, 0.20]),
            "track_status": rng.choice(track_status, rows, p=[0.77, 0.14, 0.04, 0.05]),
            "session_type": rng.choice(session_type, rows, p=[0.86, 0.14]),
        }
    )

    noise = rng.normal(0, 260, rows)
    df["lap_time_ms"] = (
        df["sector1_time"]
        + df["sector2_time"]
        + df["sector3_time"]
        + 12 * (df["tyre_life"])
        + 18 * (df["fuel_load"])
        + 8 * (df["track_temp"] - 35)
        + noise
    )

    position_signal = (
        0.18 * (df["speed_mean"] - 230)
        - 0.12 * (df["lap_time_ms"] - df["lap_time_ms"].mean()) / 100
        - 0.25 * df["brake_ratio"]
        + rng.normal(0, 1.2, rows)
    )

    ranks = pd.qcut(position_signal.rank(method="first"), 3, labels=["outside_points", "points", "podium"])
    df["result_class"] = ranks.astype(str)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
