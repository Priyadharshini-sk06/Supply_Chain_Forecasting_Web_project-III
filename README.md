# Project 4: Supply Chain Demand Forecasting

video Link:https://drive.google.com/file/d/1Q9pdL_sZJtMwKqSw26OBli5iPO8yyOly/view?usp=sharing
Basic/intermediate-level implementation of "Project 4: Supply Chain
Demand Forecasting" from the internship projects guide.

Forecasts product demand per SKU using **XGBoost** with lag /
rolling-window / calendar feature engineering, evaluated with **WMAPE**
(Weighted Mean Absolute Percentage Error), and explained with **SHAP**
feature importance. Ships with two datasets — a real one (default) and
a synthetic one — see below.

## What's implemented

- **Data**:
  - `data/download_real_data.py` (default/recommended) downloads
    **real** sales figures for 2 SKUs from a public GitHub-hosted
    dataset repository — see **"About the real dataset"** below for
    exactly what's in it and its limitations.
  - `data/generate_data.py` procedurally generates a synthetic
    multi-SKU **daily** sales dataset (20 SKUs across 4 categories, 2
    years of daily history) with the guide's ingredients: trend,
    weekly + annual seasonality, promotions, holidays, price, and
    noise. Kept as an alternative — see **"Using the synthetic
    dataset"** below.
- **Model**: `model/train.py` — feature engineering (lags, rolling
  mean/std, calendar features) feeding an `XGBRegressor`, evaluated
  per-SKU on a held-out time-based split against a naive "last period"
  baseline. A single `--freq {daily,monthly}` flag switches between
  the two datasets' feature sets (see **"How daily vs. monthly
  works"**) — same script, same backend, no other code changes.
- **Explainability**: SHAP `TreeExplainer` computes global mean
  |SHAP value| per feature, exposed via `/feature-importance`.
- **Backend**: `backend/app.py` — FastAPI service with `/forecast`
  (multi-period iterative forecast for a given SKU), `/skus`,
  `/feature-importance`, and `/health`. It reads the frequency/feature
  config that `train.py` saves, so it works unchanged against either
  dataset.
- **Frontend**: `frontend/` — a plain HTML/CSS/JS page with a SKU
  selector, a horizon slider (labeled in days or months depending on
  which model is loaded), a Chart.js line chart (actuals vs forecast),
  and a feature-importance bar chart. Served directly by the FastAPI
  backend.

## About the real dataset — please read before presenting this

The guide names M5/Walmart and Kaggle's Favorita dataset — both need a
Kaggle account, and in this development environment, network access
was restricted to GitHub and package registries (Kaggle, Hugging Face,
UCI, etc. weren't reachable). So this project uses two **real, public**
monthly sales series that are hosted directly on GitHub, from a widely
used statistics/forecasting teaching repository:

- **Shampoo Sales** — 3 years of real monthly sales for a shampoo
  company (36 data points).
- **Quebec Car Sales** — real monthly car sales in Quebec, Jan 1960 -
  Dec 1968 (108 data points).

Honest caveats:

- Both series are **monthly**, not daily — there's no real day-level
  retail dataset reachable from here, so `train.py` switches its whole
  feature set (lags of 1/2/3/6/12 months, 3/6/12-month rolling stats,
  month/quarter calendar features) when given `--freq monthly`, instead
  of the daily lag/day-of-week/promo/price features used for the
  synthetic dataset.
- There are only **2 real series** here, not the guide's 500+ SKUs, and
  they're real sales figures for two unrelated products (a shampoo
  brand and a car market) pressed into this project's "multi-SKU"
  shape — not two SKUs from the same retailer.
- No price, promotion, or holiday data exists for these series, so
  those features simply aren't used when training on real data (you'll
  see a shorter feature list in the training output).
- With only 36-108 data points per series, WMAPE numbers here are
  noisier than they'd be on a large multi-SKU retail dataset — treat
  them as representative of "a small real dataset," not as what a full
  M5/Favorita-scale run would give you.

Measured result on this real data: **14.2% WMAPE** vs. a **20.2%**
naive baseline (29.8% improvement) — within the guide's 15-20% target
range, though on a much smaller/simpler dataset than intended.

If you get the chance to download a real daily multi-SKU dataset
yourself (e.g. Kaggle's "Store Item Demand Forecasting Challenge" or
M5), see **"Using a larger real daily dataset"** below — it's a
one-command swap back to the richer `daily` feature set already built
for the synthetic data.

## Results

| Dataset | Frequency | Test WMAPE | Baseline WMAPE | Improvement |
|---|---|---|---|---|
| Real (Shampoo Sales + Quebec Car Sales) | monthly | 14.2% | 20.2% | 29.8% |
| Synthetic (20 SKUs, procedurally generated) | daily | 9.2% | 26.1% | 64.8% |

Numbers shift a little each time you retrain (small dataset, random
seeds) — treat these as representative, not exact.

## Setup & run (real dataset — default)

```bash
cd project4_supply_chain_forecasting
pip install -r requirements.txt

# 1. Download the real dataset (2 real monthly SKUs)
python data/download_real_data.py

# 2. Train the model (~5-10s on CPU)
python model/train.py --data ../data/demand_data_real.csv --freq monthly --out ./artifacts

# 3. Start the API + frontend
cd backend
uvicorn app:app --reload --port 8004
```

Then open **http://localhost:8004**, pick a SKU (`SHAMPOO-CO` or
`CAR-SALES-QC`), set a horizon (in months), and click "Forecast".

### Using the synthetic dataset instead

```bash
python data/generate_data.py --n-skus 20 --days 730
python model/train.py --data ../data/demand_data.csv --freq daily --out ./artifacts
```

Both write to the same `model/artifacts/` folder the backend reads
from, so whichever you trained most recently is what `/forecast`
serves — no other changes needed.

### API

```
GET /health                                -> status, freq ("daily"|"monthly"), held-out test metrics
GET /skus                                  -> [{"sku_id": "SHAMPOO-CO", "category": "Personal Care"}, ...]
GET /feature-importance                    -> top-10 SHAP feature importances
GET /forecast?sku_id=SHAMPOO-CO&horizon=6  -> {
      "sku_id": "SHAMPOO-CO", "category": "Personal Care", "freq": "monthly", "horizon": 6,
      "recent_actuals": [{"date": "2021-07", "sales": 812.0}, ...],
      "forecast": [{"date": "2022-01", "forecast": 1004.5}, ...],
      "model_wmape_pct": 14.2
    }
```

The forecast is generated iteratively, one period at a time: each
predicted value is fed back in as a "known" past value so lag/rolling
features stay valid further into the horizon (standard recursive
multi-step forecasting). For the daily/synthetic model, future
`promo_flag` is assumed 0 unless you extend the API to accept a promo
calendar, and `is_holiday` is computed from a fixed calendar; the
monthly/real model has no such fields.

## How daily vs. monthly works

`train.py --freq {daily,monthly}` selects a config (lags, rolling
windows, calendar columns) and auto-detects whether `price`,
`promo_flag`, `is_holiday` exist in the input CSV — using them as
features only if present. That full config is saved into
`model/artifacts/metrics.json`, and `backend/app.py` reads it back
rather than hardcoding assumptions, so the same backend code serves
either model without modification.

## Using a larger real daily dataset (recommended if grading weighs accuracy/scale)

Any CSV with these columns works with `--freq daily` and zero code
changes:

```
date, sku_id, category, sales, price, promo_flag, is_holiday
```

(`price`, `promo_flag`, `is_holiday` are optional — omit columns you
don't have and they're simply not used as features.)

1. Download Kaggle's "Store Item Demand Forecasting Challenge" dataset
   (or M5) — free account, a couple of minutes.
2. Reshape/rename its columns to match the layout above (a short
   pandas script, or ask for one).
3. Train against it directly:

```bash
python model/train.py --data /path/to/your_sales_history.csv --freq daily --out ./artifacts
```

With a real multi-SKU daily dataset you'd get both the guide's
intended data shape and (typically) a lower/more stable WMAPE than
either dataset shipped here.

## Architecture

```
Browser (frontend/, Chart.js) --HTTP--> FastAPI (backend/app.py)
                                            |-- loads XGBoost model (model/artifacts/model.joblib)
                                            |-- loads per-SKU history tail for feature building
                                            |-- loads freq/feature config + SHAP importances (model/artifacts/metrics.json)
```

## How this maps to the guide's upgrade path

- **Basic Level**: Random Forest / feature engineering — superseded
  directly by the XGBoost model below.
- **Intermediate Level**: XGBoost for tabular demand forecasting, WMAPE
  evaluation — done.
- **Advanced/SOTA**: Temporal Fusion Transformer, MLOps auto-retraining
  pipeline, and multi-task/graph-based forecasting are out of scope for
  this submission's depth — noted as a natural "next step" in the
  write-up.
