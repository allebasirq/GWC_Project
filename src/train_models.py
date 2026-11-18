"""Train and compare the three MVP binary-classification models.

Run from the repository root:
    python src/train_models.py

The test split is never used here. It is reserved for final evaluation.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import binary_labels, clean_flows, load_feature_list  # noqa: E402

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VAL_PATH = ROOT / "data" / "processed" / "val.csv"
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
LABELS = ["BENIGN", "ATTACK"]


def load_split(path: Path, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/data_pipeline.py to build the data splits."
        )

    frame = pd.read_csv(path)
    if "Label" not in frame.columns:
        raise ValueError(f"{path} is missing the required 'Label' column.")

    labels = binary_labels(frame["Label"].astype(str).str.strip())
    return clean_flows(frame, features), labels


def evaluate(name: str, pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    predictions = pipeline.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=LABELS).ravel()

    return {
        "model": name,
        "evaluation_split": "validation",
        "validation_rows": int(len(y_val)),
        "attack_recall": float(tp / (tp + fn)),
        "false_alarms": int(fp),
        "false_alarm_rate": float(fp / (fp + tn)),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_charts(
    pipeline,
    features: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Save the four chart files expected by app/app.py."""
    attack_probability = pipeline.predict_proba(X_val)[:, list(pipeline.classes_).index("ATTACK")]

    cm = confusion_matrix(y_val, pipeline.predict(X_val), labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=LABELS)
    ax.set_yticks([0, 1], labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(column, row, f"{cm[row, column]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_val, attack_probability, pos_label="ATTACK"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(false_positive_rate, true_positive_rate, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(
        y_val, attack_probability, pos_label="ATTACK"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    classifier = pipeline
    if hasattr(pipeline, "named_steps"):
        classifier = pipeline.named_steps["logisticregression"]

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = abs(classifier.coef_[0])
    else:
        raise ValueError("The selected model does not expose feature importance.")

    ranked = sorted(zip(features, importance), key=lambda item: item[1], reverse=True)[:15]
    names = [item[0] for item in ranked][::-1]
    values = [item[1] for item in ranked][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values)
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Contributions")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    features = load_feature_list()
    X_train, y_train = load_split(TRAIN_PATH, features)
    X_val, y_val = load_split(VAL_PATH, features)

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Training and validation splits must contain both classes.")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }

    scores = []
    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        score = evaluate(name, pipeline, X_val, y_val)
        scores.append(score)
        print(
            f"{name}: attack recall {score['attack_recall']:.1%}, "
            f"false alarms {score['false_alarms']:,} "
            f"({score['false_alarm_rate']:.1%})"
        )

    # Missing attacks are more serious, so choose the highest recall first.
    # Use the lowest false-alarm rate to break ties.
    final_name = min(
        scores,
        key=lambda score: (
            -score["attack_recall"],
            score["false_alarm_rate"],
        ),
    )["model"]
    final_score = next(score for score in scores if score["model"] == final_name)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipeline in trained.items():
        joblib.dump(
            {"pipeline": pipeline, "features": features, "labels": LABELS},
            MODELS_DIR / f"{name}.joblib",
        )

    joblib.dump(
        {"pipeline": trained[final_name], "features": features, "labels": LABELS},
        MODELS_DIR / "binary_model.joblib",
    )
    save_charts(trained[final_name], features, X_val, y_val)

    metrics = {
        **final_score,
        "positive_label": "ATTACK",
        "models": scores,
        "final_model": final_name,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nSelected final model: {final_name}")
    print(f"Saved final model: models/binary_model.joblib")
    print(f"Saved model comparison: models/metrics.json")


if __name__ == "__main__":
    main()
