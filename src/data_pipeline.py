"""
Data pipeline for the Cybersecurity Threat Detection System.

Two parts, matching docs/interface_contract.md:

1. clean_flows(df)   SHARED by training and src/inference.py.
                      Keeps every row (so results line up with the uploaded
                      file) and returns exactly the columns in
                      models/feature_list.json, in that order.

2. build_dataset()   TRAINING ONLY. Removes broken and repeated rows, drops
                      useless columns, samples, splits, and saves
                      models/feature_list.json.

Build the data (run from the repo root):
    python src/data_pipeline.py \
        data/raw/Tuesday-WorkingHours.pcap_ISCX.csv \
        data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
        --sample-frac 0.10

Outputs:
    data/processed/clean_sample.csv, train.csv, val.csv, test.csv
    models/feature_list.json

The output files keep the `Label` column (BENIGN, DDoS, FTP-Patator,
SSH-Patator). Use binary_labels() to get BENIGN / ATTACK for the binary model.
There is no separate target column, so nothing can leak into the features.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURE_LIST_PATH = ROOT / "models" / "feature_list.json"
LABEL_COL = "Label"

# These identify one specific connection instead of describing its behavior.
# A model could memorize them, so they are never features.
ID_COLUMNS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]


# ---------------------------------------------------------------------------
# Shared by training and inference
# ---------------------------------------------------------------------------

def load_feature_list(path=FEATURE_LIST_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the data build step first "
            "(python src/data_pipeline.py ...)."
        )
    return json.loads(path.read_text())


def clean_flows(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Prepare flows for the model. Keeps every row and its index.

    Raises ValueError (with a readable message) if required columns are missing,
    so the website can show it to the user.
    """
    if features is None:
        features = load_feature_list()

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()  # headers can have stray spaces

    missing = [c for c in features if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(
            f"This file is missing {len(missing)} required column(s): {shown}"
        )

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)  # broken values -> 0
    return X


def binary_labels(labels: pd.Series) -> pd.Series:
    """BENIGN stays BENIGN; every attack type becomes ATTACK."""
    return labels.where(labels == "BENIGN", "ATTACK")


# ---------------------------------------------------------------------------
# Training only
# ---------------------------------------------------------------------------

def _load_raw(path) -> pd.DataFrame:
    # latin-1 avoids decode errors that some CICIDS2017 files trigger
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  loaded {Path(path).name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def _drop_repeated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Some files list 'Fwd Header Length' twice (pandas names the copy '.1')."""
    drop = []
    for c in df.columns:
        base = re.sub(r"\.\d+$", "", c)
        if base != c and base in df.columns and df[c].equals(df[base]):
            drop.append(c)
    return df.drop(columns=drop)


def _training_clean(df: pd.DataFrame):
    """Row-removing cleanup. Never use this on uploaded files."""
    n0 = len(df)
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    df = _drop_repeated_columns(df)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    feats = [c for c in df.columns if c != LABEL_COL]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n1 = len(df)

    df = df.drop_duplicates()
    n2 = len(df)

    constant = [c for c in feats if df[c].nunique() <= 1]
    df = df.drop(columns=constant)

    print(f"  rows: {n0:,} -> {n1:,} after removing broken values "
          f"-> {n2:,} after removing repeats")
    print(f"  dropped {len(constant)} constant columns: {constant}")

    features = [c for c in df.columns if c != LABEL_COL]
    return df.reset_index(drop=True), features


def build_dataset(inputs, sample_frac=0.10, out_dir=None, feature_list_path=None,
                  seed=42):
    out = Path(out_dir) if out_dir else ROOT / "data" / "processed"
    fl_path = Path(feature_list_path) if feature_list_path else FEATURE_LIST_PATH
    out.mkdir(parents=True, exist_ok=True)
    fl_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    frames = [_load_raw(p) for p in inputs]
    common = set.intersection(*(set(f.columns) for f in frames))
    df = pd.concat([f[[c for c in f.columns if c in common]] for f in frames],
                   ignore_index=True)

    print("Cleaning...")
    df, features = _training_clean(df)

    print(f"Sampling {sample_frac:.0%}...")
    df = (df.groupby(LABEL_COL, group_keys=False)
            .sample(frac=sample_frac, random_state=seed)
            .reset_index(drop=True))
    print("Class counts in sample:")
    print(df[LABEL_COL].value_counts().to_string())

    # 70% practice / 15% check / 15% exam, same attack ratio in each.
    train, rest = train_test_split(df, test_size=0.30, stratify=df[LABEL_COL],
                                   random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[LABEL_COL],
                                 random_state=seed)

    df.to_csv(out / "clean_sample.csv", index=False)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    fl_path.write_text(json.dumps(features, indent=2))

    print("\nAttack share per part (should be nearly identical):")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        share = (part[LABEL_COL] != "BENIGN").mean()
        print(f"  {name:5s} {len(part):>8,} rows, {share:.1%} attacks")
    print(f"\nSaved data to {out.resolve()}")
    print(f"Saved {len(features)} feature names to {fl_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the cleaned, split dataset.")
    ap.add_argument("inputs", nargs="+", help="raw CICIDS2017 CSV files")
    ap.add_argument("--sample-frac", type=float, default=0.10)
    ap.add_argument("--out-dir", default=None, help="default: data/processed")
    ap.add_argument("--feature-list", default=None,
                    help="default: models/feature_list.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_dataset(args.inputs, args.sample_frac, args.out_dir,
                  args.feature_list, args.seed)
