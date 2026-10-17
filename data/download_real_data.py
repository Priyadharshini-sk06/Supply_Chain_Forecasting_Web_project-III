"""
Build a REAL demand dataset (not synthetic) for the Supply Chain Demand
Forecasting project, from two classic, openly-hosted real sales time
series published in Jason Brownlee's Datasets repo on GitHub (no login
needed):

  - shampoo.csv          -> 3 years of monthly shampoo sales for a company
  - monthly-car-sales.csv -> Quebec monthly car sales, Jan 1960 - Dec 1968

Why these two: this environment's network access is restricted to
GitHub and package registries -- Kaggle (which hosts the guide's named
datasets: M5/Walmart, Favorita) and similar hosts are not reachable
here. These two series are real, publicly documented sales figures
that are reachable, so we use them and are upfront about the trade-off
below rather than silently staying synthetic.

Important honesty notes:
  - Both series are MONTHLY, not daily. The guide's spec (daily
    multi-horizon forecasting, promotions, holidays, competitor
    pricing) assumes daily retail data, which isn't available to fetch
    here. model/train.py auto-detects monthly vs. daily data (via
    --freq) and adjusts its feature engineering accordingly (lag
    1/2/3/6/12 months and 3/6/12-month rolling stats, instead of daily
    lags) -- there is no price/promo/holiday signal in this real data,
    so those features are simply not used when training on it.
  - There are only 2 real series here (36 and 108 monthly data points),
    not the guide's 500+ SKUs. This is a real but small dataset -- WMAPE
    on such a short series is noisier than on a large multi-SKU retail
    dataset. See README.md for how to swap in a full real dataset
    (e.g. Kaggle's Store Item Demand Forecasting Challenge) if you get
    a chance to download one yourself.

Usage:
    python download_real_data.py --out ../data/demand_data_real.csv
"""
import argparse
import io
import os

import pandas as pd
import requests

SHAMPOO_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
CAR_SALES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"


def load_shampoo():
    resp = requests.get(SHAMPOO_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # original "Month" column is like "1-01", "1-02", ... "3-12" (year-in-series, not a
    # real calendar year) -- assign a synthetic but consistent monthly calendar so we
    # have real dates to compute month/quarter features from. The SALES figures are real.
    dates = pd.date_range("2019-01-01", periods=len(df), freq="MS")
    return pd.DataFrame({
        "date": dates,
        "sku_id": "SHAMPOO-CO",
        "category": "Personal Care",
        "sales": df["Sales"].astype(float),
    })


def load_car_sales():
    resp = requests.get(CAR_SALES_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    dates = pd.to_datetime(df["Month"], format="%Y-%m")
    return pd.DataFrame({
        "date": dates,
        "sku_id": "CAR-SALES-QC",
        "category": "Automotive",
        "sales": df["Sales"].astype(float),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "demand_data_real.csv"))
    args = ap.parse_args()

    print("Fetching real sales series from GitHub (jbrownlee/Datasets) ...")
    shampoo = load_shampoo()
    cars = load_car_sales()
    df = pd.concat([shampoo, cars], ignore_index=True).sort_values(["sku_id", "date"])
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} rows (2 real SKUs) to {args.out}")
    print(df.groupby("sku_id")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
