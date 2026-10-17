"""
Synthetic multi-SKU daily demand dataset generator for the Supply Chain
Demand Forecasting project.

Real datasets referenced in the guide (M5 / Walmart - 42,000 series,
Favorita - Kaggle, ~125M rows) require Kaggle credentials and are far
larger than needed for a course project. This script generates a
smaller but realistic multi-SKU daily sales dataset with the same
structural ingredients called out in the guide: trend, weekly +
annual seasonality, promotions, holidays, and noise -- so the modeling
code (feature engineering, XGBoost/LSTM, SHAP) faces the same kind of
signal it would on a real retail dataset. Swap in a real CSV with the
same columns to use real data.

Output: data/demand_data.csv with columns:
    date, sku_id, category, sales, price, promo_flag, is_holiday
"""
import argparse
import os

import numpy as np
import pandas as pd

CATEGORIES = ["Beverages", "Snacks", "Household", "Personal Care"]

HOLIDAYS_BY_MONTH_DAY = {
    (1, 1), (1, 26), (3, 8), (8, 15), (10, 2), (10, 24), (11, 12), (12, 25),
}


def make_series(sku_id, category, n_days, start_date, rng):
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    base_level = rng.uniform(20, 200)
    trend = np.linspace(0, rng.uniform(-15, 40), n_days)

    weekly = 1 + 0.35 * np.sin(2 * np.pi * (dates.dayofweek.values) / 7 + rng.uniform(0, 2 * np.pi))
    annual = 1 + 0.5 * np.sin(2 * np.pi * (dates.dayofyear.values) / 365.25 + rng.uniform(0, 2 * np.pi))

    price = rng.uniform(3, 25)
    price_series = price * (1 + rng.normal(0, 0.02, n_days).cumsum() * 0.01)
    price_series = np.clip(price_series, price * 0.7, price * 1.3)

    promo_flag = (rng.random(n_days) < 0.08).astype(int)
    is_holiday = np.array([1 if (d.month, d.day) in HOLIDAYS_BY_MONTH_DAY else 0 for d in dates])

    promo_lift = 1 + promo_flag * rng.uniform(0.3, 0.9)
    holiday_lift = 1 + is_holiday * rng.uniform(0.2, 0.6)
    price_elasticity = (price / price_series) ** rng.uniform(0.5, 1.5)

    noise = rng.normal(1, 0.08, n_days)

    demand = (base_level + trend) * weekly * annual * promo_lift * holiday_lift * price_elasticity * noise
    demand = np.clip(demand, 0, None)
    demand = np.round(demand).astype(int)

    return pd.DataFrame({
        "date": dates,
        "sku_id": sku_id,
        "category": category,
        "sales": demand,
        "price": np.round(price_series, 2),
        "promo_flag": promo_flag,
        "is_holiday": is_holiday,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "demand_data.csv"))
    ap.add_argument("--n-skus", type=int, default=20)
    ap.add_argument("--days", type=int, default=730)  # 2 years
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = []
    for i in range(args.n_skus):
        sku_id = f"SKU-{i + 1:03d}"
        category = CATEGORIES[i % len(CATEGORIES)]
        frames.append(make_series(sku_id, category, args.days, args.start_date, rng))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({args.n_skus} SKUs x {args.days} days) to {args.out}")
    print(df.groupby("sku_id")["sales"].agg(["mean", "std"]).head())


if __name__ == "__main__":
    main()
