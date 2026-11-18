"""
Upload and Results page (owner: Nicole).

Run from the repo root:
    streamlit run app/app.py

Other pages (such as Varushi's Replay page) go in app/pages/ and appear in the
sidebar automatically. They can reuse `predict` from src/inference.py.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inference import BENIGN, InputError, load_model, predict, read_flow_file, summarize  # noqa: E402

MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
CHARTS = [
    ("confusion_matrix.png", "Confusion matrix: what the model got right and wrong"),
    ("roc_curve.png", "ROC curve"),
    ("pr_curve.png", "Precision-recall curve"),
    ("feature_importance.png", "Which clues the model relies on most"),
]
# Columns that help a person investigate a flagged row, shown first when present.
KEY_COLUMNS = ["Source IP", "Destination IP", "Destination Port", "Timestamp",
               "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
               "Flow Bytes/s", "Label"]

st.set_page_config(page_title="Threat Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


# ---------------------------------------------------------------------------
# Upload and Results
# ---------------------------------------------------------------------------

def show_upload_and_results():
    try:
        model = get_model()
    except (FileNotFoundError, ValueError) as err:
        st.error(f"The model isn't ready yet. {err}")
        return

    uploaded = st.file_uploader("Choose a CSV file of network flows", type=["csv"])
    if uploaded is None:
        st.info("No file yet. To try it out, upload one of the sample files from "
                "the `data/samples/` folder.")
        return

    try:
        with st.spinner("Checking each connection..."):
            results = predict(read_flow_file(uploaded), model)
    except InputError as err:
        st.error(f"**We couldn't use this file.** {err}")
        return
    except Exception as err:  # anything unexpected: keep the page running
        st.error(f"Something went wrong while checking this file: {err}")
        return

    counts = summarize(results)

    # --- Summary ----------------------------------------------------------
    st.subheader(f"Results for {uploaded.name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Connections checked", f"{counts['total']:,}")
    c2.metric("Normal", f"{counts['normal']:,}")
    c3.metric("Flagged as attack", f"{counts['attack']:,}",
              f"{counts['attack_share']:.1%} of the file", delta_color="off")

    chart = pd.DataFrame({"Connections": [counts["normal"], counts["attack"]]},
                         index=["Normal", "Attack"])
    st.bar_chart(chart)

    if counts["attack"] == 0:
        st.success("No connections in this file look like an attack.")
    else:
        st.warning(f"{counts['attack']:,} connection(s) look like an attack. "
                   "A person should still investigate each one.")

    if "Label" in results.columns:
        show_label_check(results)

    # --- Flagged rows -----------------------------------------------------
    flagged = results[results["prediction"].astype(str) != BENIGN].copy()
    if flagged.empty:
        return

    st.subheader("Flagged connections")
    flagged.insert(0, "Row in file", flagged.index + 1)
    flagged["Attack chance"] = flagged["probability"] * 100
    flagged = flagged.sort_values("Attack chance", ascending=False)

    show_all = st.checkbox("Show all columns")
    if show_all:
        cols = (["Row in file", "Attack chance", "prediction"]
                + [c for c in flagged.columns
                   if c not in ("Row in file", "Attack chance", "prediction", "probability")])
    else:
        cols = (["Row in file", "Attack chance", "prediction"]
                + [c for c in KEY_COLUMNS if c in flagged.columns])

    st.dataframe(
        flagged[cols],
        hide_index=True,
        width="stretch",
        column_config={
            "Attack chance": st.column_config.ProgressColumn(
                "Attack chance", format="%.0f%%", min_value=0, max_value=100),
            "prediction": "Verdict",
        },
    )
    st.download_button(
        "Download flagged rows (CSV)",
        flagged.drop(columns=["Attack chance"]).to_csv(index=False),
        file_name=f"flagged_{uploaded.name}",
        mime="text/csv",
    )


def show_label_check(results):
    """If the file already says which rows are attacks, compare with the model."""
    truth_attack = results["Label"].astype(str).str.strip() != BENIGN
    said_attack = results["prediction"].astype(str) != BENIGN
    caught = int((truth_attack & said_attack).sum())
    missed = int((truth_attack & ~said_attack).sum())
    false_alarms = int((~truth_attack & said_attack).sum())
    real_attacks = int(truth_attack.sum())

    with st.expander("This file has true labels, so we can check the model on it"):
        a, b, c = st.columns(3)
        a.metric("Attacks caught", f"{caught:,} of {real_attacks:,}")
        b.metric("Missed attacks", f"{missed:,}")
        c.metric("False alarms", f"{false_alarms:,}")
        st.caption("Missed attacks are the most serious mistake: real attacks the "
                   "model called normal.")


# ---------------------------------------------------------------------------
# How good is the model?
# ---------------------------------------------------------------------------

def show_model_quality():
    if not METRICS_PATH.exists():
        st.info("Scores will appear here once Isabella saves `models/metrics.json`.")
        return

    m = json.loads(METRICS_PATH.read_text())
    comparisons = m.get("models")
    if comparisons:
        st.subheader("Model comparison")
        comparison = pd.DataFrame(comparisons)
        comparison["attack_recall"] *= 100
        comparison["false_alarm_rate"] *= 100
        comparison = comparison.rename(columns={
            "model": "Model",
            "attack_recall": "Attack recall",
            "false_alarms": "False alarms",
            "false_alarm_rate": "False-alarm rate",
            "false_negatives": "Missed attacks",
        })
        comparison = comparison[[
            "Model", "Attack recall", "False alarms",
            "False-alarm rate", "Missed attacks",
        ]]
        st.dataframe(
            comparison,
            hide_index=True,
            column_config={
                "Attack recall": st.column_config.NumberColumn(format="%.1f%%"),
                "False-alarm rate": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption(
            f"Final model selected: **{str(m.get('final_model', m.get('model'))).replace('_', ' ')}**"
        )

    tp, fn = m.get("true_positives"), m.get("false_negatives")
    tn, fp = m.get("true_negatives"), m.get("false_positives", m.get("false_alarms"))

    model_name = str(m.get("model", "model")).replace("_", " ")
    split = m.get("evaluation_split", "test")
    st.write(
        f"Scores for the **{model_name}** model on the {split} rows. "
        "These rows were not used to train the model, although they were used "
        "to compare models and select the final model."
    )

    if None not in (tp, fn, tn, fp):
        a, b, c = st.columns(3)
        a.metric("Attacks caught", f"{tp / (tp + fn):.1%}", f"{tp:,} of {tp + fn:,}",
                 delta_color="off")
        b.metric("Missed attacks", f"{fn:,}", "most serious mistake", delta_color="off")
        c.metric("False alarms", f"{fp:,}", f"{fp / (fp + tn):.1%} of normal traffic",
                 delta_color="inverse")

        st.markdown("**What the model said vs. the truth**")
        st.table(pd.DataFrame(
            {"Model said normal": [f"{tn:,}", f"{fn:,} (missed)"],
             "Model said attack": [f"{fp:,} (false alarm)", f"{tp:,}"]},
            index=["Really normal", "Really an attack"],
        ))
    else:
        st.json(m)

    charts = [(MODELS_DIR / name, caption) for name, caption in CHARTS
              if (MODELS_DIR / name).exists()]
    if charts:
        cols = st.columns(2)
        for i, (path, caption) in enumerate(charts):
            cols[i % 2].image(str(path), caption=caption, width="stretch")
    else:
        st.caption("Charts will appear here once Isabella saves them to `models/`.")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("🛡️ Cybersecurity Threat Detection System")
st.write("Upload a CSV file of **recorded** network flows (CICIDS2017 format). "
         "The model checks each connection and flags the ones that look like "
         "an attack (DDoS or brute-force password guessing).")
st.caption("For defense and learning only. A flag means a connection *looks like* "
           "an attack; a person should always investigate before acting.")

tab_results, tab_quality = st.tabs(["Upload and Results", "How good is the model?"])
with tab_results:
    show_upload_and_results()
with tab_quality:
    show_model_quality()
