"""
Make the small sample files for the Replay page (owner: Varushi)

Takes rows from the TEST split (rows the model never studies) and saves 
three small files in data/samples/. The keep the Label column, so the
website can compare what really happened with what the model said.

Run from the repo root, after daya_pipeline.py:
    python src/make_samples.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data"/ "processed" / "test.csv"
SAMPLES_DIR = ROOT / "data" / "samples"
SEED = 42

# What goes in each sample file
SAMPLES = {
    "normal.csv": [("BENIGN", 300)],
    "ddos.csv": [("BENIGN", 150), ("DDoS", 350)],
    "brute_force.csv": [("BENIGN", 150), ("FTP-Patator", 175), ("SSH-Patator", 175)]
}


def take(test, label, n):
    """Pick up to n random rows with this label."""
    rows = test[test["Label"] == label]
    return rows.sample(n=min(n, len(rows)), random_state=SEED)


def main():
    test = pd.read_csv(TEST_PATH)
    test["Label"] = test["Label"].astype(str).str.strip()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for name, parts in SAMPLES.items():
        pieces = [take(test, label, n) for label, n in parts]
        #Shuffle so the attacks are mixed in with normal traffic
        sample = pd.concat(pieces).sample(frac=1, random_state=SEED)
        path = SAMPLES_DIR / name
        sample.to_csv(path, index=False)

        counts = sample["Label"].value_counts().to_dict()
        size_mb = path.stat().st_size / 1e6
        print(f"Saved {name}: {len(sample)} rows, {size_mb:.2f} MB, {counts}")


if __name__ == "__main__":
    main()
