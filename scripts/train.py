#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.f1_predictions.pipeline import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lap-time and race-result models from telemetry feature data."
    )
    parser.add_argument("--input", required=True, help="Path to telemetry feature CSV.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/models",
        help="Directory to write trained models and metadata.",
    )
    parser.add_argument(
        "--metrics-file",
        default="artifacts/models/metrics.json",
        help="Where to store evaluation metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    metrics_file = Path(args.metrics_file)

    df = pd.read_csv(input_path)
    bundle, metrics = train_models(df)
    bundle.save(output_dir)

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
