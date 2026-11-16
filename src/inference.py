"""
Inference for the Cybersecurity Threat Detection System.

Turns an uploaded file of network flows into a verdict for every row.
Follows docs/interface_contract.md:

    predict(df)  ->  the same rows, plus two new columns:
        prediction    BENIGN or ATTACK (or an attack type for the Stretch model)
        probability   the model's estimated chance (0 to 1) that the row is an attack

The website (app/app.py) and the Replay page both use this file, so the model
is loaded and the data is cleaned the same way everywhere.

Quick check from the repo root:
    python src/inference.py path/to/flows.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import clean_flows  # noqa: E402  (shared cleaning step)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "binary_model.joblib"
BENIGN = "BENIGN"


class InputError(ValueError):
    """A problem with the user's file. The message is safe to show on the website."""


def load_model(path=MODEL_PATH) -> dict:
    """Load the saved model dictionary: {"pipeline", "features", "labels"}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"The model file {path.name} was not found in the models folder. "
            "Train it with `python src/train_logistic_regression.py`, "
            "or ask Isabella for the file."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "features" not in bundle:
        raise ValueError(
            f"{path.name} is not in the expected format "
            '({"pipeline": ..., "features": [...], "labels": [...]}).'
        )
    return bundle


def read_flow_file(file) -> pd.DataFrame:
    """Read an uploaded CSV (a path or a file object) into a DataFrame.

    Raises InputError with a plain-language message if the file can't be used.
    """
    try:
        # latin-1 matches data_pipeline.py and never fails on odd characters
        df = pd.read_csv(file, encoding="latin-1", low_memory=False)
    except pd.errors.EmptyDataError:
        raise InputError("This file is empty. Please choose a CSV file of network flows.")
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise InputError(
            "This file could not be read as a CSV. Make sure it is a comma-separated "
            "file exported from CICIDS2017 or a similar flow tool."
        )

    df.columns = df.columns.astype(str).str.strip()
    # After removing spaces, two headers can end up with the same name; keep the first.
    df = df.loc[:, ~df.columns.duplicated()]

    if df.empty:
        raise InputError("This file has column names but no rows of data.")
    return df


def predict(df: pd.DataFrame, model: dict | None = None) -> pd.DataFrame:
    """Return df with `prediction` and `probability` columns added.

    Raises InputError if required columns are missing.
    """
    bundle = model if model is not None else load_model()
    try:
        X = clean_flows(df, bundle["features"])
    except ValueError as err:  # clean_flows explains which columns are missing
        raise InputError(
            f"{err}. The file needs the CICIDS2017 flow columns "
            "(for example 'Flow Duration' and 'Total Fwd Packets')."
        ) from err

    pipeline = bundle["pipeline"]
    results = df.copy()
    results["prediction"] = pipeline.predict(X)

    # Chance of attack = 1 - chance of BENIGN. This works for both the binary
    # model (BENIGN / ATTACK) and the Stretch multi-class model.
    classes = list(getattr(pipeline, "classes_", []))
    if hasattr(pipeline, "predict_proba") and BENIGN in classes:
        proba = pipeline.predict_proba(X)
        results["probability"] = 1.0 - proba[:, classes.index(BENIGN)]
    else:
        results["probability"] = np.nan
    return results


def summarize(results: pd.DataFrame) -> dict:
    """Counts for the website: total rows, normal rows and flagged rows."""
    is_attack = results["prediction"].astype(str) != BENIGN
    total = len(results)
    attacks = int(is_attack.sum())
    return {
        "total": total,
        "normal": total - attacks,
        "attack": attacks,
        "attack_share": attacks / total if total else 0.0,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inference.py path/to/flows.csv")
        sys.exit(1)
    out = predict(read_flow_file(sys.argv[1]))
    s = summarize(out)
    print(f"{s['total']:,} rows: {s['normal']:,} normal, {s['attack']:,} flagged "
          f"({s['attack_share']:.1%})")
