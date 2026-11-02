# Cybersecurity Threat Detection System

A club project that trains a machine-learning model to spot attacks in **recorded** network traffic, with a small website for exploring the results.

**Presentation:** Friday, November 13, 2026

## What this project is

Network traffic is recorded as "flows" (one row per connection between two computers). We train a model on the public CICIDS2017 dataset to tell **normal** traffic from **attacks** (DDoS and brute-force password guessing). A website lets someone upload a file of flows, or play a sample, and see which connections were flagged.

**What it is not:** a live monitor of an internet connection. It analyzes recorded data and is designed for analysts and learners. It is for **defense only** and contains no attack tools.

## MVP and Stretch goals

**MVP (must work)**
- Binary detection: normal vs. attack
- Two CICIDS2017 daily files (one with brute-force attacks, one with DDoS), using a 5 to 10% sample
- Cleaning: fix broken values, remove repeated rows and useless columns
- Fair split: practice, check and exam parts with the same attack ratio in each
- Models: logistic regression, decision tree, random forest
- Honest scoring: attacks caught, false alarms, missed attacks
- Website with two pages: **Upload and Results** and **Replay**
- Write-up and slides covering findings, limits and ethics

**Stretch (only if ahead of schedule)**
- Multi-class: say which attack it is (DDoS, FTP-Patator, SSH-Patator) and show an attack-type chart
- A fourth model (XGBoost or LightGBM)
- Explainability page (which features matter most)
- Test on a different day's traffic
- Replay that reveals rows one at a time

**If we fall behind, cut in this order:** row-by-row replay, different-day test, Explainability page, fourth model, attack-type chart.

## Team

| Name | Role | Owns |
|---|---|---|
| Isabella | Project manager and model lead | Repo setup, starter code, training and scoring the models, saved model |
| Varushi | Data and Replay page | Data download, cleaning, fair split, sample files, Replay page |
| Nicole | Website and upload code | `inference.py`, upload checks, Upload and Results page |

## Website pages

| Page | Owner | What it does |
|---|---|---|
| Upload and Results | Nicole | Upload a flow CSV, see flagged rows, a normal vs. attack count, and model scores and charts |
| Replay | Varushi | Pick a sample (normal, DDoS, brute force) and see what gets flagged |
| Explainability (Stretch) | Varushi | Shows which features matter most to the model |

## Repository structure

```
data/
  raw/         Original CICIDS2017 CSVs (not tracked by Git)
  processed/   Cleaned data and splits (not tracked by Git)
  samples/     Small sample files for Replay (tracked by Git)
docs/          Team agreements (interface_contract.md)
notebooks/     Jupyter notebooks for exploration and training
src/           Shared code: data_pipeline.py, inference.py
models/        Saved model, scores (JSON) and charts (PNG)
app/           Streamlit website (app.py and pages/)
```

## Getting started

**You need:** Python 3.10 or newer, Git, and VS Code (recommended).

1. Clone the repo:
```
   git clone https://github.com/allebasirq/GWC_Project.git
   cd GWC_Project
```
2. Create and activate a virtual environment.

   Mac or Linux:
```
   python3 -m venv .venv
   source .venv/bin/activate
```
   Windows (PowerShell):
```
   python -m venv .venv
   .venv\Scripts\activate
```
3. Install the libraries:
```
   pip install -r requirements.txt
```
4. Add the data (see below).
5. Run the website:
```
   streamlit run app/app.py
```
   It opens at http://localhost:8501.

## Data

We use the **CICIDS2017 "MachineLearningCVE"** CSV files (flow features, not raw packet captures), published by the Canadian Institute for Cybersecurity. Search for "CICIDS2017 MachineLearningCVE" to find the official download page.

Download only the two daily files with the attacks we need and place them in `data/raw/`. The `data/raw/` and `data/processed/` folders are ignored by Git because the files are very large. Small sample files live in `data/samples/` and are committed. Do not commit full datasets.

## Team workflow

- Nobody pushes directly to `main`.
- Create your own branch (for example `nicole/upload-page`), do your work there, and open a pull request.
- A teammate reviews each pull request before it is merged.
- Check-ins: Mondays, 15 minutes. Each person shares what they finished, what is next, and what is blocking them.
- Keep `requirements.txt` up to date when you add a library.
- Keep each file in `data/samples/` small (a few MB at most).

## Team agreements

Details are in `docs/interface_contract.md`. In short:
- **Labels:** `BENIGN` and `ATTACK` (MVP). Stretch adds `DDoS`, `FTP-Patator`, `SSH-Patator`.
- **Shared cleaning:** one function, `clean_flows(df)`, in `src/data_pipeline.py`, used by both training and the website.
- **Model file:** `models/binary_model.joblib` (MVP) and `models/multiclass_model.joblib` (Stretch).
- **Prediction:** `predict(df)` in `src/inference.py` returns the rows plus `prediction` and `probability` columns.
- **Saved outputs for the website:** `metrics.json` and chart images in `models/`.

## Schedule

| Week | Dates | Focus |
|---|---|---|
| 1 | Oct 19 to 25 | Repo setup, starter code, data download and cleaning, website frame |
| 2 | Oct 26 to Nov 1 | Fair split, three models, upload code, Replay layout |
| 3 | Nov 2 to 8 | Final model (Nov 4), real model in the website, testing |
| 4 | Nov 9 to 13 | Freeze (Nov 10), write-up, slides, backup video, rehearsal, presentation |

**Key dates:** repo ready Oct 20, clean data Oct 25, final model Nov 4, freeze Nov 10, presentation Nov 13.

## Limits and ethics

- Our data comes from a lab, so real networks may behave differently.
- Similar rows may still overlap between the practice and exam parts, so scores can look better than they would in practice.
- Missing a real attack (a false negative) is the most serious mistake, so we report it clearly.
- The model says a connection *looks like* an attack. A person must still investigate it.
- Only record networks you own or have permission to monitor.
- This project is for defense only.

## Status

- [x] Repo and environment set up
- [ ] Starter code shared
- [ ] Data cleaned and split
- [ ] Three models trained and scored
- [ ] Final model saved
- [ ] Upload and Results page working
- [ ] Replay page working
- [ ] Write-up and slides complete