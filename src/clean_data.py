"""
Clean, sample and split CICIDS2017 CSV files.

Usage (from the project folder):
    python clean_data.py \
        data/raw/Tuesday-WorkingHours.pcap_ISCX.csv \
        data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
        --sample-frac 0.10 --out-dir data/clean

Outputs (in --out-dir):
    clean_sample.csv      the cleaned sample (all parts together)
    train.csv, val.csv, test.csv   practice / check / exam parts
    feature_columns.json  the exact feature names the model expects
                          (Nicole can use this to validate uploads)

Rerunnable: same inputs + same seed = same outputs.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Columns that identify a specific connection rather than describe its behavior.
# Keeping them lets a model "cheat" by memorizing addresses, so we remove them.
ID_COLUMNS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]


def load_one(path: str) -> pd.DataFrame:
    # latin-1 avoids decode errors some CICIDS2017 files trigger with utf-8
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()  # headers have stray leading spaces
    print(f"  loaded {Path(path).name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def drop_repeated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Some files contain 'Fwd Header Length' twice (pandas names the copy '.1')."""
    drop = [c for c in df.columns
            if re.search(r"\.\d+$", c)
            and re.sub(r"\.\d+$", "", c) in df.columns
            and df[c].equals(df[re.sub(r"\.\d+$", "", c)])]
    return df.drop(columns=drop)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    df = drop_repeated_columns(df)

    # Tidy labels, then keep the original attack name and add a 0/1 target.
    df["Label"] = df["Label"].astype(str).str.strip()

    # Broken values: 'Infinity' / 'inf' strings and NaN in the rate columns.
    feature_cols = [c for c in df.columns if c != "Label"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n1 = len(df)

    # Repeated rows.
    df = df.drop_duplicates()
    n2 = len(df)

    # Useless columns: the same value in every row.
    constant = [c for c in feature_cols if c in df.columns and df[c].nunique() <= 1]
    df = df.drop(columns=constant)

    print(f"  rows: {n0:,} -> {n1:,} after removing broken values "
          f"-> {n2:,} after removing repeats")
    print(f"  dropped {len(constant)} constant columns: {constant}")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="raw CSV files")
    ap.add_argument("--sample-frac", type=float, default=0.10)
    ap.add_argument("--out-dir", default="data/clean")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    frames = [load_one(p) for p in args.inputs]
    common = set.intersection(*(set(f.columns) for f in frames))
    df = pd.concat([f[[c for c in f.columns if c in common]] for f in frames],
                   ignore_index=True)

    print("Cleaning...")
    df = clean(df)

    # Sample, keeping the attack mix (BENIGN, DDoS, FTP-Patator, SSH-Patator).
    print(f"Sampling {args.sample_frac:.0%}...")
    df = (df.groupby("Label", group_keys=False)
            .sample(frac=args.sample_frac, random_state=args.seed)
            .reset_index(drop=True))

    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)
    print("Class counts in sample:")
    print(df["Label"].value_counts().to_string())

    # 70% practice / 15% check / 15% exam, same attack ratio in each.
    train, rest = train_test_split(df, test_size=0.30, stratify=df["Label"],
                                   random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["Label"],
                                 random_state=args.seed)

    df.to_csv(out / "clean_sample.csv", index=False)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    features = [c for c in df.columns if c not in ("Label", "is_attack")]
    (out / "feature_columns.json").write_text(json.dumps(features, indent=2))

    print("\nAttack share per part (should be nearly identical):")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s} {len(part):>8,} rows, {part['is_attack'].mean():.1%} attacks")
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
