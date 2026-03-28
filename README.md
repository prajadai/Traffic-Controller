# Fuzzy Logic Traffic Light Controller

A Streamlit-based data science mini project that uses a Mamdani fuzzy inference system to compute adaptive green-light durations from real-time traffic conditions.

## Project Overview

This project models an intelligent traffic signal controller using two inputs:

- Traffic density (vehicles per minute)
- Waiting time (seconds)

The controller outputs:

- Green light duration (seconds)

The app focuses on explainability, not just prediction. It shows membership functions, active rules, rule firing strengths, and defuzzification output so you can understand the decision process end-to-end.

## Main Files

- no-main.py
  - Current primary Streamlit app (matplotlib-based visualizations)
  - Includes tabs for rules, live control, membership functions, rule activation, and scenarios

- requirements.txt
  - Python dependency list used to set up the project environment

- .gitignore
  - Ignore rules for virtual environments, caches, local secrets, and generated artifacts

## Core Working Logic

The fuzzy controller is implemented in the FuzzyTrafficController class.

### 1) Membership Function Definitions

Three linguistic variables are defined:

1. Traffic Density (0 to 60)
   - Low (trapezoidal)
   - Medium (triangular)
   - High (trapezoidal)

2. Waiting Time (0 to 90)
   - Short (trapezoidal)
   - Medium (triangular)
   - Long (trapezoidal)

3. Green Duration (0 to 60)
   - Short (triangular)
   - Medium (triangular)
   - Long (triangular)
   - Very Long (trapezoidal)

Triangular and trapezoidal functions are used because they are simple, interpretable, and common in Mamdani systems.

### 2) Fuzzification

Crisp input values are converted to membership degrees in [0, 1].

Example idea:
- A density value can be partly Medium and partly High at the same time.

### 3) Rule Evaluation

There are 9 IF-THEN rules combining density and waiting categories.
Rule firing strength is computed with:

- AND operator = min(mu_density, mu_waiting)

Only rules with firing strength greater than 0 contribute to the output.

### 4) Aggregation + Defuzzification

For each fired rule:

- The output membership function is clipped at rule strength.
- All clipped outputs are aggregated.
- Final green time is computed by center of gravity (COG):

COG = (integral x * mu(x) dx) / (integral mu(x) dx)

In code, this is approximated numerically over the output range using a small step.

## App Tabs (no-main.py)

1. Fuzzy Rules
   - Conceptual explanation and complete rule base

2. Live Controller
   - Interactive sliders for density and waiting
   - Real-time fuzzy memberships and computed green time
   - Defuzzification formula and output plot

3. Membership Functions
   - Input/output membership curves
   - Current input marker and degree visualization

4. Rule Activation
   - Fired rules and their activation strengths

5. Scenarios
   - Preset traffic situations (rush hour, late night, etc.)
   - One-click loading into the live controller

## Libraries Used

- streamlit
  - Web app UI and interaction

- numpy
  - Numerical operations and defuzzification sampling

- pandas
  - Rule table display and lightweight tabular handling

- matplotlib
  - Membership plots, rule activation charts, and traffic-light drawing

- warnings (standard library)
  - Suppresses warning noise for cleaner app output

## Setup and Run

### 1) Create and activate a virtual environment (recommended)

Python 3.10+ is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

This installs all dependencies used by the project.

### 3) Run the main app

```powershell
streamlit run .\no-main.py
```

If streamlit is not on PATH, run:

```powershell
python -m streamlit run .\no-main.py
```

### 4) Open in browser

Streamlit will print a local URL such as:

http://localhost:8501

## Current Project Structure

```text
AI-Project/
├── fuzzy-traffic-controller.py
├── README.md
├── requirements.txt
├── .gitignore
└── __pycache__/
```

## Troubleshooting

- Error: "streamlit is not recognized"
  - Use python -m streamlit run .\no-main.py
  - Or ensure the virtual environment is activated before running

- Session state issues
  - This app uses Streamlit session state to synchronize sliders and scenario presets
  - Restart the app if state appears stale after code edits

- Missing package import errors
  - Install dependencies in the active environment
  - Verify interpreter selection in VS Code

## Suggested Next Improvements

- Add unit tests for membership and defuzzification functions
- Add data logging of scenarios and outputs for analysis
- Add sensitivity analysis for rule changes
