# Fuzzy Logic Traffic Light Controller

An interactive Streamlit app that implements a Mamdani fuzzy inference system to compute adaptive green-light duration from traffic conditions.

## What This Project Does

The controller takes two inputs:

- Traffic density (vehicles/minute)
- Waiting time (seconds)

It outputs one control value:

- Recommended green-light duration (seconds)

The app is explainable by design: it visualizes membership functions, fired rules, clipping/aggregation, centroid defuzzification, scenario behavior, and response-surface metrics.

## Current App Features

The Streamlit app in `fuzzy-traffic-controller.py` currently has 6 tabs:

1. Fuzzy Rules
2. Live Controller
3. Rule Activation
4. Defuzzification Walkthrough
5. Preset Scenarios
6. Evaluation Metrics

### Inference Pipeline Implemented

1. Fuzzification
   - Converts crisp density/waiting inputs to membership degrees.

2. Rule Evaluation
   - Uses 9 IF-THEN rules.
   - Rule strength uses AND = min(mu_density, mu_waiting).

3. Aggregation
   - Clips each output set by rule strength.
   - Aggregates using pointwise max over fired-rule outputs.

4. Defuzzification
   - Uses centroid (center of gravity) over the aggregated output.
   - Numerically sampled over output domain.

## Membership Variables and Ranges

- Traffic Density: 0 to 60
  - Low (trapezoidal), Medium (triangular), High (trapezoidal)

- Waiting Time: 0 to 90
  - Short (trapezoidal), Medium (triangular), Long (trapezoidal)

- Green Duration: 0 to 60
  - Short (triangular), Medium (triangular), Long (triangular), Very Long (trapezoidal)


## Setup

Python 3.10+ recommended.

### 1) Create and activate virtual environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the app

```powershell
streamlit run .\fuzzy-traffic-controller.py
```

If `streamlit` is not on PATH:

```powershell
python -m streamlit run .\fuzzy-traffic-controller.py
```

### 4) Open in browser

Streamlit typically serves at:

http://localhost:8501

## Project Files

- `fuzzy-traffic-controller.py`: Main application and fuzzy controller logic
- `requirements.txt`: Dependency list
- `README.md`: Project documentation

## Troubleshooting

- Command not found for Streamlit:
  - Use `python -m streamlit run .\fuzzy-traffic-controller.py`
  - Confirm virtual environment is activated

- Import errors:
  - Re-run `pip install -r requirements.txt` in the active environment
  - Verify VS Code interpreter selection points to your environment

- Session state feels stale after code edits:
  - Refresh the app page or restart Streamlit