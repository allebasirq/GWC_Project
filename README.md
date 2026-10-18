# Cybersecurity Threat Detection System

Machine-learning system that detects benign vs. attack network traffic (DDoS, FTP-Patator, SSH-Patator) using the CICIDS2017 dataset, with a Streamlit dashboard for inspecting flagged flows.

**Presentation:** Friday, November 13, 2026

## Project overview

We train and compare machine-learning classifiers on network-flow data, then wrap the best model in a dashboard. A user can upload a flow CSV or replay sample traffic and see flagged flows, an attack-type breakdown, and model metrics.

**Scope**
- Binary detection first (benign vs. attack), then multi-class (DDoS, FTP-Patator, SSH-Patator)
- Flow-feature CSVs only, not raw PCAPs
- Detection only, for defensive use

## Team

| Name | Role | Owns |
|---|---|---|
| Isabella | Mentor, ML lead | Repo setup, scope, ensemble models, imbalance handling, evaluation, stretch test, saved model |
| Varushi | Backend (data) and frontend | Data pipeline, cleaning, splits, baselines; Replay and Explainability pages |
| Nicole | Backend (inference) and frontend | `inference.py`, upload validation, app shell; Upload and Results and Model Performance pages |

## Dashboard pages

| Page | Owner | Backed by |
|---|---|---|
| App shell and navigation | Nicole | Streamlit |
| Upload and Results | Nicole | `src/inference.py` |
| Model Performance | Nicole | Saved metrics and plots in `models/` |
| Replay | Varushi | Sample files in `data/samples/` |
| Explainability | Varushi | SHAP or feature importance on the final model |

## Repository structure

```
data/
  raw/         Original CICIDS2017 CSVs (not tracked by Git)
  processed/   Cleaned data and splits (not tracked by Git)
  samples/     Small replay sample files (tracked by Git)
notebooks/     Jupyter notebooks for exploration
src/           Final pipeline modules (.py): data_pipeline.py, inference.py, etc.
models/        Saved model (model.joblib), metrics (JSON) and plots (PNG)
app/           Streamlit dashboard (app.py and pages)
```

## Getting started

**Requirements:** Python 3.10 or newer, Git, VS Code (recommended).

1. Clone the repo:
```
   git clone https://github.com/allebasirq/GWC_Project.git
   cd GWC_Project
```
2. Create and activate a virtual environment.

   Mac or Linux:
```
   python3 -m venv venv
   source venv/bin/activate
```
   Windows (PowerShell):
```
   python -m venv venv
   venv\Scripts\activate
```
3. Install dependencies:
```
   pip install -r requirements.txt
```
4. Download the data (see below).
5. Run the dashboard:
```
   streamlit run app/app.py
```
   It opens at http://localhost:8501.

**Mac note:** if XGBoost fails with a `libomp` error, run `brew install libomp`.

## Data

We use the **CICIDS2017 "MachineLearningCVE"** CSV files (flow features, not raw PCAPs), published by the Canadian Institute for Cybersecurity. Search for "CICIDS2017 MachineLearningCVE" to find the official download page.

Place the CSV files in `data/raw/`. The `data/raw/` and `data/processed/` folders are ignored by Git because the files are very large. Small replay samples live in `data/samples/` and are committed. Do not commit full datasets.

## Team workflow

- Never push directly to `main`. Create a branch (for example `nicole/results-page`) and open a pull request.
- Another team member reviews each pull request before merging.
- Check-ins: Tuesday and Friday, 15 minutes.
- Keep `requirements.txt` up to date when you add a library.
- Keep each file in `data/samples/` small (a few MB at most).

## Interface contract

- **Labels:** `BENIGN`, `DDoS`, `FTP-Patator`, `SSH-Patator`
- **Model file:** one joblib file in `models/` containing the full pipeline (scaler and model) plus the feature list
- **Feature columns:** exact list and order defined in `src/` (finalized in Week 1)
- **Saved outputs for the dashboard:** metrics as JSON and plots as PNG in `models/`

## Schedule

| Week | Dates | Focus |
|---|---|---|
| 1 | Oct 19 to 25 | Repo setup, data loading and cleaning, app shell, dashboard wireframe, interface contract |
| 2 | Oct 26 to Nov 1 | Splits, baselines, ensemble models, inference module, page layouts |
| 3 | Nov 2 to 8 | Evaluation, final model (by Nov 4), all five dashboard pages with the real model, stretch test |
| 4 | Nov 9 to 13 | Feature freeze (Nov 10), write-up, slides, backup demo, rehearsal, presentation |

**Key deadlines:** repo ready Oct 20, cleaned data Oct 25, final model Nov 4, feature freeze Nov 10, presentation Nov 13.

## Ethics and limitations

- This system is for **detection only** and defensive use. It contains no attack tooling.
- CICIDS2017 is lab-generated and may not reflect real-world networks.
- Results may drop on traffic from other days or environments.
- Missed attacks (false negatives) are the most serious error, so we report attack recall prominently.
- Uploaded traffic data should be treated as sensitive.

## Status

- [x] Repo and environment set up
- [ ] Data cleaned and split
- [ ] Baseline models trained
- [ ] Final model saved
- [ ] Dashboard pages working
- [ ] Write-up and slides complete