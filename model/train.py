"""
Train an XGBoost demand forecasting model (intermediate level of the
guide: "Upgrade to XGBoost for tabular demand forecasting") with lag /
rolling / calendar feature engineering, evaluated with WMAPE, plus a
SHAP explainer for feature-importance driven explanations.

Supports two data frequencies via --freq:
  - daily   (default) -- e.g. the synthetic multi-SKU dataset from
    generate_data.py, with day-of-week/holiday/promo/price signal.
  - monthly -- e.g. the real dataset from download_real_data.py, which
    only has date/sku_id/category/sales (no promo/price/holiday, and
    calendar features are month/quarter instead of day-of-week).

Extra columns (price, promo_flag, is_holiday) are auto-detected from
the input CSV and only used as features when present, so the same
script and the same FastAPI backend work unchanged on either dataset.

Usage:
    python train.py --data ../data/demand_data.csv --freq daily --out ./artifacts
    python train.py --data ../data/demand_data_real.csv --freq monthly --out ./artifacts_real
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

FREQ_CONFIG = {
    "daily": {
        "lags": [1, 7, 14, 28],
        "rolling_windows": [7, 14, 28],
        "calendar_cols": ["dayofweek", "day", "month", "weekofyear", "is_weekend"],
        "default_test_periods": 60,
    },
    "monthly": {
        "lags": [1, 2, 3, 6, 12],
        "rolling_windows": [3, 6, 12],
        "calendar_cols": ["month", "quarter"],
        "default_test_periods": 6,
    },
}
OPTIONAL_COLS = ["price", "promo_flag", "is_holiday"]


def add_calendar_features(df, freq):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_lag_and_rolling_features(df, lags, rolling_windows):
    df = df.sort_values(["sku_id", "date"])
    g = df.groupby("sku_id")["sales"]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in rolling_windows:
        df[f"rollmean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        df[f"rollstd_{w}"] = g.shift(1).rolling(w).std().reset_index(level=0, drop=True)
    return df


def build_feature_frame(df, freq):
    cfg = FREQ_CONFIG[freq]
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_calendar_features(df, freq)
    df = add_lag_and_rolling_features(df, cfg["lags"], cfg["rolling_windows"])
    df["sku_code"] = df["sku_id"].astype("category").cat.codes
    df["category_code"] = df["category"].astype("category").cat.codes
    df = df.dropna().reset_index(drop=True)
    return df


def build_feature_cols(freq, available_cols):
    cfg = FREQ_CONFIG[freq]
    optional_present = [c for c in OPTIONAL_COLS if c in available_cols]
    return (
        [f"lag_{l}" for l in cfg["lags"]]
        + [f"rollmean_{w}" for w in cfg["rolling_windows"]]
        + [f"rollstd_{w}" for w in cfg["rolling_windows"]]
        + cfg["calendar_cols"]
        + optional_present
        + ["sku_code", "category_code"]
    )


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def per_sku_time_split(df, test_periods):
    """Take the last `test_periods` rows of each SKU's own history as test,
    the rest as train -- works for any frequency since it's row-based, and
    keeps every SKU represented in both splits."""
    df = df.sort_values(["sku_id", "date"])
    test_idx = df.groupby("sku_id").tail(test_periods).index
    test = df.loc[test_idx]
    train = df.drop(test_idx)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "demand_data.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "artifacts"))
    ap.add_argument("--freq", choices=["daily", "monthly"], default="daily")
    ap.add_argument("--test-periods", type=int, default=None,
                     help="most recent periods per SKU held out for testing "
                          "(default: 60 for daily, 6 for monthly)")
    args = ap.parse_args()

    cfg = FREQ_CONFIG[args.freq]
    test_periods = args.test_periods or cfg["default_test_periods"]

    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(args.data)
    feature_cols = build_feature_cols(args.freq, raw.columns)
    feat = build_feature_frame(raw, args.freq)
    train_df, test_df = per_sku_time_split(feat, test_periods)

    X_train, y_train = train_df[feature_cols], train_df["sales"]
    X_test, y_test = test_df[feature_cols], test_df["sales"]

    print(f"Freq: {args.freq}  Feature columns: {feature_cols}")
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    metrics = {
        "wmape_pct": wmape(y_test.values, preds),
        "mae": mean_absolute_error(y_test, preds),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_periods": test_periods,
        "freq": args.freq,
    }

    # naive baseline (previous-period value) for comparison, matching guide's
    # "15-20% reduction in WMAPE vs baseline" target framing
    baseline_preds = test_df["lag_1"].values
    metrics["baseline_wmape_pct"] = wmape(y_test.values, baseline_preds)
    metrics["wmape_improvement_pct"] = (
        (metrics["baseline_wmape_pct"] - metrics["wmape_pct"]) / metrics["baseline_wmape_pct"] * 100
    )

    print(f"Model WMAPE: {metrics['wmape_pct']:.2f}%  "
          f"Baseline (lag-1) WMAPE: {metrics['baseline_wmape_pct']:.2f}%  "
          f"Improvement: {metrics['wmape_improvement_pct']:.1f}%")

    # SHAP global feature importance
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(500, len(X_test)), random_state=42) if len(X_test) else X_test
    shap_values = explainer.shap_values(sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: -x[1]
    )

    # persist artifacts
    joblib.dump(model, os.path.join(args.out, "model.joblib"))
    sku_categories = raw[["sku_id", "category"]].drop_duplicates().sort_values("sku_id")
    sku_categories.to_csv(os.path.join(args.out, "sku_categories.csv"), index=False)

    # keep enough recent history per SKU so the API can build features for
    # future periods without re-reading the full raw CSV
    optional_present = [c for c in OPTIONAL_COLS if c in raw.columns]
    history_cols = ["date", "sku_id", "category", "sales"] + optional_present
    lookback = max(cfg["lags"] + cfg["rolling_windows"]) + 5
    history_tail = (
        raw[history_cols]
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values(["sku_id", "date"])
        .groupby("sku_id", as_index=False)
        .tail(lookback)
    )
    history_tail.to_csv(os.path.join(args.out, "history_tail.csv"), index=False)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({
            "metrics": metrics,
            "feature_importance": importance,
            "feature_cols": feature_cols,
            "config": {
                "freq": args.freq,
                "lags": cfg["lags"],
                "rolling_windows": cfg["rolling_windows"],
                "calendar_cols": cfg["calendar_cols"],
                "optional_cols_used": optional_present,
            },
        }, f, indent=2)

    print(f"Saved model + metrics + SHAP importance to {args.out}")
    print("Top 5 features by mean |SHAP|:")
    for name, val in importance[:5]:
        print(f"  {name}: {val:.3f}")


if __name__ == "__main__":
    main()
