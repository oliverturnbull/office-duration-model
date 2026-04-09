# Office Duration Model

Estimates how long each person spends in the office from access card tap-in/tap-out data, handling missing tap-outs (e.g. tailgating) using survival analysis.

## Approach

Missing tap-outs are treated as **right-censored** observations. A Cox proportional hazards model estimates the probability of departure over time, conditioned on:

- Person identity
- Day of week
- Arrival hour

For censored sessions the model predicts expected duration; for complete sessions it uses the observed duration.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare your data

Provide a CSV with columns:

| Column      | Description                          |
|-------------|--------------------------------------|
| `person_id` | Unique identifier for each person    |
| `timestamp` | ISO 8601 datetime of the tap event   |
| `direction` | `"in"` or `"out"`                    |

### 2. Run the pipeline

```bash
# Build sessions from raw taps, fit model, and produce estimates
python -m office_duration.cli data/taps.csv --output results.csv
```

### 3. Output

`results.csv` contains one row per session:

| Column              | Description                                    |
|---------------------|------------------------------------------------|
| `person_id`         | Person identifier                              |
| `tap_in`            | Session start time                             |
| `tap_out`           | Observed tap-out (NaT if missing)              |
| `censored`          | Whether the tap-out was missing                |
| `observed_hours`    | Duration from complete pair (null if censored)  |
| `estimated_hours`   | Model-estimated duration for all sessions       |

### 4. Generate synthetic data for testing

```bash
python -m office_duration.generate_data --people 50 --days 30 --tailgate-rate 0.2 --output data/taps.csv
```

## Project structure

```
office_duration/
├── __init__.py
├── cli.py              # Command-line entry point
├── generate_data.py    # Synthetic data generator
├── sessions.py         # Pair taps into sessions
├── features.py         # Feature engineering
└── model.py            # Survival model fitting & prediction
```
