# F1-predictions

End-to-end F1 telemetry ML starter pipeline for predicting:
- **Lap times** (regression)
- **Race result band** (`podium`, `points`, `outside_points`) (classification)

This repository now includes a lightweight training pipeline that you can run on engineered telemetry features exported by your data processing workflow.

## Project structure

- `src/f1_predictions/pipeline.py` - model training, evaluation, and artifact serialization.
- `scripts/train.py` - CLI entrypoint for training on a CSV file.
- `scripts/generate_sample_data.py` - generates synthetic telemetry features for local testing.
- `requirements.txt` - Python dependencies.

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare your telemetry feature dataset

Create a CSV at minimum with these columns:

### Numeric features
- `speed_mean`, `speed_std`, `throttle_mean`, `brake_ratio`, `rpm_mean`, `gear_changes`
- `sector1_time`, `sector2_time`, `sector3_time`
- `track_temp`, `air_temp`, `humidity`, `wind_speed`
- `fuel_load`, `tyre_life`

### Categorical features
- `team`, `driver`, `compound`, `track_status`, `session_type`

### Targets
- `lap_time_ms`
- Either:
  - `result_class` (`podium` / `points` / `outside_points`), or
  - `final_position` (will be mapped into those classes)

> Tip: if you do not have data yet, generate a synthetic sample:

```bash
python scripts/generate_sample_data.py
```

## 3) Train models

```bash
python scripts/train.py --input data/sample_telemetry_features.csv --output-dir artifacts/models --metrics-file artifacts/models/metrics.json
```

Outputs:
- `artifacts/models/lap_time_model.joblib`
- `artifacts/models/result_model.joblib`
- `artifacts/models/feature_columns.joblib`
- `artifacts/models/metrics.json`

## 4) Integrate with real team telemetry

For real-world use, your pipeline should transform raw telemetry (`speed`, `throttle`, `brake`, `rpm`, weather, tyre stint, fuel estimates, sector splits, etc.) into lap-level features matching the schema above.

Possible next improvements:
1. Replace tree models with XGBoost/LightGBM/CatBoost for better tabular performance.
2. Add temporal context (previous-lap features, stint windows, degradation trend).
3. Train separate models per circuit family (street, high-downforce, power tracks).
4. Add uncertainty estimation (quantile regression / conformal prediction).
5. Evaluate by race weekend split to avoid leakage.

## Example metrics (synthetic dataset)

On the included synthetic data generation process, you should observe high score quality (because signal is engineered into the fake dataset). Expect lower metrics on real telemetry until feature engineering and leakage controls are tightened.
