"""
Replay page (owner: Varushi)

Pick a sample of recorded traffic and see which connections the model flags.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from inference import BENIGN, load_model, predict, summarize

SAMPLES_DIR = ROOT / "data" / "samples"
SAMPLES = {
    "Normal traffic": "normal.csv",
    "DDoS attack": "ddos.csv",
    "Brute-force password guessing (FTP and SSH)": "brute_force.csv",
}

st.set_page_config(page_title="Replay", page_icon="▶️", layout="wide")
st.title("▶️ Replay")
st.write("Pick a sample of **recorded** traffic and see which connections "
         "the model flags as an attack.")


@st.cache_resource
def get_model():
    return load_model()


choice = st.selectbox("Choose a sample", list(SAMPLES))
path = SAMPLES_DIR / SAMPLES[choice]

if not path.exists():
    st.info(f"The sample file {path.name} isn't in data/samples/ yet. "
            "Run python src/make_samples.py to make it.")
    st.stop()

try:
    model = get_model()
except (FileNotFoundError, ValueError) as err:
    st.error(f"The model isn't ready yet. {err}")
    st.stop()

flows = pd.read_csv(path)
flows.columns = flows.columns.str.strip()
results = predict(flows, model)
counts = summarize(results)

# What the model said
st.subheader("What the model said")
c1, c2, c3 = st.columns(3)
c1.metric("Connections", f"{counts['total']:,}")
c2.metric("Normal", f"{counts['normal']:,}")
c3.metric("Flagged as attack", f"{counts['attack']:,}")

# Compare with the truth
really_attack = results["Label"].astype(str).str.strip() != BENIGN
said_attack = results["prediction"].astype(str) != BENIGN

st.subheader("Was the model right?")
a, b, c = st.columns(3)
a.metric("Attacks caught", f"{(really_attack & said_attack).sum():,} "
                           f"of {really_attack.sum():,}")
b.metric("Missed attacks", f"{(really_attack & ~said_attack).sum():,}")
c.metric("False alarms", f"{(~really_attack & said_attack).sum():,}")
st.caption("Missed attacks are the most serious mistake: real attacks "
           "the model called normal.")

# Every connection
st.subheader("Every connection")
only_flagged = st.checkbox("Show only flagged connections")

table = pd.DataFrame({
    "Row": results.index + 1,
    "Really": results["Label"],
    "Model said": results["prediction"],
    "Attack chance": results["probability"] * 100,
    "Correct?": (really_attack == said_attack).map({True: "✅", False: "❌"}),
})
if only_flagged:
    table = table[said_attack.values]

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Attack chance": st.column_config.ProgressColumn(
            "Attack chance", format="%.0f%%", min_value=0, max_value=100)
    },
)  
st.caption("For defense and learning only. A flag means a connection *looks like* "
           "an attack; a person should always investigate before acting.")
