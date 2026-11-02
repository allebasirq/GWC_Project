# Interface Contract
Changes need agreement from all three of us.

## Labels
- Binary: `BENIGN`, `ATTACK`
- Multi-class: `BENIGN`, `DDoS`, `FTP-Patator`, `SSH-Patator`
- All other attack types are excluded.

## Feature columns
- All columns of the cleaned data except `Label`, in a fixed order.
- Saved by `src/data_pipeline.py` to `models/feature_list.json`.
- Final list: TBD after the first cleaned sample (Oct 25).

## Preprocessing
- One shared function `clean_flows(df)` in `src/data_pipeline.py`.
- Used by both training and `src/inference.py`.

## Model files (in `models/`)
- `binary_model.joblib`, `multiclass_model.joblib`
- Each is a dictionary: `{"pipeline": ..., "features": [...], "labels": [...]}`

## Inference
- `src/inference.py` provides `predict(df)`.
- Input: DataFrame of flows. Output: same rows plus `prediction` and `probability` columns.

## Saved outputs for the dashboard (in `models/`)
- `metrics.json`, `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `feature_importance.png`

## Folders
- `data/` (raw, processed, samples), `notebooks/`, `src/`, `models/`, `app/`