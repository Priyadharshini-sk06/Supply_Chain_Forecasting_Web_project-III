"""
FastAPI backend for the Supply Chain Demand Forecasting project.

Works unchanged against either the daily synthetic dataset or the real
monthly dataset -- it reads the frequency/feature config that
model/train.py saves into metrics.json rather than hardcoding it.

Endpoints:
    GET  /health                        -> service + model status
    GET  /skus                          -> list of available SKU ids + categories
    GET  /forecast?sku_id=...&horizon=N -> N-period forecast + recent actuals + SHAP feature importance
                                            (a "period" is a day for the daily/synthetic model,
                                             a month for the real monthly model -- see /health "freq")

Run:
    uvicorn app:app --reload --port 8004
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="Supply Chain Demand Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
history_tail = None
sku_categories = None
metrics = {}
config = {}
sku_code_map = {}
category_code_map = {}

HOLIDAYS = {
    (1, 1), (1, 26), (3, 8), (8, 15), (10, 2), (10, 24), (11, 12), (12, 25),
}


@app.on_event("startup")
def load_artifacts():
    global model, history_tail, sku_categories, metrics, config, sku_code_map, category_code_map

    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    if not os.path.exists(model_path):
        print(f"WARNING: model not found at {model_path}. Run model/train.py first.")
        return

    model = joblib.load(model_path)
    history_tail = pd.read_csv(os.path.join(ARTIFACTS_DIR, "history_tail.csv"), parse_dates=["date"])
    sku_categories = pd.read_csv(os.path.join(ARTIFACTS_DIR, "sku_categories.csv"))

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
        metrics.update(json.load(f))
    config.update(metrics.get("config", {"freq": "daily", "lags": [1, 7, 14, 28],
                                          "rolling_windows": [7, 14, 28],
                                          "calendar_cols": ["dayofweek", "day", "month", "weekofyear", "is_weekend"],
                                          "optional_cols_used": ["price", "promo_flag", "is_holiday"]}))

    # NOTE: matches train.py's `astype("category").cat.codes`, which assigns
    # codes in sorted (alphabetical) order of the category's unique values.
    skus_sorted = sorted(sku_categories["sku_id"].unique())
    sku_code_map = {s: i for i, s in enumerate(skus_sorted)}
    cats_sorted = sorted(sku_categories["category"].unique())
    category_code_map = {c: i for i, c in enumerate(cats_sorted)}
    print(f"Model + history loaded. freq={config.get('freq')}")


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "freq": config.get("freq", "daily"),
        "metrics": metrics.get("metrics"),
        "n_skus": int(sku_categories["sku_id"].nunique()) if sku_categories is not None else 0,
    }


@app.get("/skus")
def list_skus():
    if sku_categories is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")
    return sku_categories.to_dict(orient="records")


@app.get("/feature-importance")
def feature_importance():
    if not metrics:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")
    return {"feature_importance": metrics.get("feature_importance", [])[:10]}


def _advance_date(date: pd.Timestamp, steps: int):
    if config.get("freq") == "monthly":
        return date + pd.DateOffset(months=steps)
    return date + pd.Timedelta(days=steps)


def _build_future_features(sku_id: str, future_date: pd.Timestamp, series: pd.DataFrame):
    """series: chronological dataframe (one row per period, no gaps) with
    columns date, sales, category, and any optional cols (price/promo_flag/
    is_holiday) -- includes any already-forecasted future periods appended.
    Lags/rolling stats are computed by ROW POSITION (matching train.py's
    groupby().shift()), which is valid because `series` has exactly one row
    per period with no missing periods."""
    row = {}
    n = len(series)
    for lag in config["lags"]:
        row[f"lag_{lag}"] = float(series["sales"].iloc[n - lag]) if n >= lag else float(series["sales"].iloc[0])

    for w in config["rolling_windows"]:
        window_vals = series["sales"].tail(w)
        row[f"rollmean_{w}"] = float(window_vals.mean())
        row[f"rollstd_{w}"] = float(window_vals.std()) if len(window_vals) > 1 else 0.0

    calendar_cols = config["calendar_cols"]
    if "dayofweek" in calendar_cols:
        row["dayofweek"] = future_date.dayofweek
    if "day" in calendar_cols:
        row["day"] = future_date.day
    if "month" in calendar_cols:
        row["month"] = future_date.month
    if "quarter" in calendar_cols:
        row["quarter"] = future_date.quarter
    if "weekofyear" in calendar_cols:
        row["weekofyear"] = int(future_date.isocalendar().week)
    if "is_weekend" in calendar_cols:
        row["is_weekend"] = int(future_date.dayofweek >= 5)

    last_known = series.iloc[-1]
    optional_cols = config.get("optional_cols_used", [])
    if "is_holiday" in optional_cols:
        row["is_holiday"] = int((future_date.month, future_date.day) in HOLIDAYS)
    if "promo_flag" in optional_cols:
        row["promo_flag"] = 0  # assume no scheduled promo unless told otherwise
    if "price" in optional_cols:
        row["price"] = float(last_known.get("price", 10.0))

    row["sku_code"] = sku_code_map.get(sku_id, 0)
    row["category_code"] = category_code_map.get(last_known.get("category"), 0)
    return row


@app.get("/forecast")
def forecast(sku_id: str = Query(...), horizon: int = Query(14, ge=1, le=90)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")

    hist = history_tail[history_tail["sku_id"] == sku_id].sort_values("date").copy()
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sku_id '{sku_id}'")

    freq = config.get("freq", "daily")
    date_fmt = "%Y-%m" if freq == "monthly" else "%Y-%m-%d"

    recent_actuals = hist[["date", "sales"]].tail(30 if freq == "daily" else len(hist)).copy()
    recent_actuals["date"] = recent_actuals["date"].dt.strftime(date_fmt)

    keep_cols = ["date", "sales", "category"] + [c for c in config.get("optional_cols_used", []) if c in hist.columns]
    series = hist[keep_cols].copy()
    last_date = series["date"].max()

    feature_cols = metrics.get("feature_cols", [])
    forecasts = []
    for step in range(1, horizon + 1):
        future_date = _advance_date(last_date, step)
        feat_row = _build_future_features(sku_id, future_date, series)
        X = pd.DataFrame([feat_row])[feature_cols]
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)
        forecasts.append({"date": future_date.strftime(date_fmt), "forecast": round(pred, 1)})

        new_row = {"date": future_date, "sales": pred, "category": series["category"].iloc[-1]}
        for c in config.get("optional_cols_used", []):
            if c in feat_row:
                new_row[c] = feat_row[c]
        series = pd.concat([series, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "sku_id": sku_id,
        "category": str(hist["category"].iloc[-1]),
        "freq": freq,
        "horizon": horizon,
        "recent_actuals": recent_actuals.to_dict(orient="records"),
        "forecast": forecasts,
        "model_wmape_pct": metrics.get("metrics", {}).get("wmape_pct"),
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
