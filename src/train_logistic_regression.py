"""Train and evaluate the binary logistic-regression baseline.

Run from the repository root:
    python src/train_logistic_regression.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import binary_labels, clean_flows, load_feature_list

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VAL_PATH = ROOT / "data" / "processed" / "val.csv"
MODEL_PATH = ROOT / "models" / "binary_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
LABELS = ["BENIGN", "ATTACK"]


def load_split(path: Path, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/data_pipeline.py to build the data splits."
        )

    frame = pd.read_csv(path)
    if "Label" not in frame.columns:
        raise ValueError(f"{path} is missing the required 'Label' column.")

    labels = frame["Label"].astype(str).str.strip()
    X = clean_flows(frame, features)
    y = binary_labels(labels)
    return X, y


def main() -> None:
    features = load_feature_list()
    X_train, y_train = load_split(TRAIN_PATH, features)
    X_val, y_val = load_split(VAL_PATH, features)

    if y_train.nunique() != 2:
        raise ValueError("The training split must contain BENIGN and attack rows.")
    if y_val.nunique() != 2:
        raise ValueError("The validation split must contain BENIGN and attack rows.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=LABELS
    ).ravel()
    attack_recall = tp / (tp + fn)
    false_alarm_rate = fp / (fp + tn)

    metrics = {
        "model": "logistic_regression",
        "evaluation_split": "validation",
        "positive_label": "ATTACK",
        "validation_rows": int(len(y_val)),
        "attack_recall": attack_recall,
        "false_alarms": int(fp),
        "false_alarm_rate": false_alarm_rate,
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": model, "features": features, "labels": LABELS},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Validation rows: {len(y_val):,}")
    print(f"Attack recall: {attack_recall:.1%} ({tp:,} of {tp + fn:,} attacks caught)")
    print(
        f"False alarms: {fp:,} of {fp + tn:,} benign rows "
        f"({false_alarm_rate:.1%})"
    )
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved validation metrics: {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
