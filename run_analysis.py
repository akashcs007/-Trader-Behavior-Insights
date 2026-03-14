"""
Trader Performance vs Market Sentiment — run from terminal:
  cd c:/Users/akash/Videos/trader_sentiment_analysis
  pip install pandas numpy matplotlib seaborn scipy openpyxl
  python run_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Optional: only needed for plots
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

DATA_DIR = Path(__file__).resolve().parent / "data"
SENTIMENT_PATH = DATA_DIR / "fear_greed_index.csv"
TRADER_PATH = DATA_DIR / "historical_trader_data.csv"


def find_data_path(base: Path, exts=(".csv", ".parquet", ".xlsx")):
    if base.exists():
        return base
    for ext in exts:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    # Handle double extension (e.g. historical_trader_data.csv.csv)
    if base.suffix == ".csv":
        double_csv = base.parent / (base.name + ".csv")
        if double_csv.exists():
            return double_csv
    # Fallback: any file in data/ whose name contains the base stem
    stem = base.stem
    for f in base.parent.iterdir():
        if f.is_file() and stem in f.name and f.suffix.lower() in (".csv", ".parquet", ".xlsx"):
            return f
    return None


def load_sentiment():
    path = find_data_path(SENTIMENT_PATH)
    if not path:
        print("ERROR: Sentiment file not found. Place fear_greed_index.csv (or .xlsx) in data/")
        sys.exit(1)
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date_only"] = df["date"].dt.date
    return df


def load_trades():
    path = find_data_path(TRADER_PATH)
    if not path:
        print("ERROR: Trader data file not found. Place historical_trader_data.csv (or .parquet/.xlsx) in data/")
        sys.exit(1)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    time_col = "time" if "time" in df.columns else "timestamp"
    if time_col in df.columns:
        ts = df[time_col]
        # Unix timestamp: if numeric and large, assume milliseconds else seconds
        if pd.api.types.is_numeric_dtype(ts):
            unit = "ms" if ts.dropna().max() > 1e12 else "s"
            df[time_col] = pd.to_datetime(ts, unit=unit, errors="coerce")
        else:
            df[time_col] = pd.to_datetime(ts, errors="coerce")
        df["date_only"] = df[time_col].dt.date
    return df, time_col


def main():
    print("Loading sentiment...")
    sentiment_df = load_sentiment()
    sentiment_col = None
    for c in ["classification", "value", "index", "fear_greed"]:
        if c in sentiment_df.columns:
            sentiment_col = c
            break
    if sentiment_col is None:
        sentiment_col = sentiment_df.columns[1]
    print("Sentiment shape:", sentiment_df.shape)

    print("Loading trader data...")
    trades_df, time_col = load_trades()
    pnl_col = None
    for c in ["closedpnl", "closed_pnl"]:
        if c in trades_df.columns:
            pnl_col = c
            break
    if pnl_col is None and any("pnl" in c.lower() for c in trades_df.columns):
        pnl_col = [c for c in trades_df.columns if "pnl" in c.lower()][0]
    if pnl_col:
        trades_df[pnl_col] = pd.to_numeric(trades_df[pnl_col], errors="coerce")
    print("Trades shape:", trades_df.shape)

    # Merge on date (ensure same type: both date objects)
    trader_date_min, trader_date_max = trades_df["date_only"].min(), trades_df["date_only"].max()
    sentiment_daily = sentiment_df.drop_duplicates(subset=["date_only"]).set_index("date_only")[[sentiment_col]].copy()
    sentiment_daily.columns = ["sentiment"]
    trades_df["date_only"] = pd.to_datetime(trades_df["date_only"], errors="coerce").dt.date
    trades_df = trades_df.merge(sentiment_daily, left_on="date_only", right_index=True, how="inner")
    print("Trades with sentiment:", len(trades_df))
    if len(trades_df) == 0:
        print("WARNING: No overlapping dates. Check that trader timestamps and sentiment dates are in the same range.")
        print("  Sentiment date range:", sentiment_df["date_only"].min(), "to", sentiment_df["date_only"].max())
        print("  Trader date range:   ", trader_date_min, "to", trader_date_max)
        sys.exit(1)

    # Sentiment regime (numeric or categorical)
    if trades_df["sentiment"].dtype in [np.float64, np.int64]:
        bins = [0, 25, 45, 55, 75, 100]
        labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        trades_df["sentiment_regime"] = pd.cut(trades_df["sentiment"], bins=bins, labels=labels, include_lowest=True)
    else:
        trades_df["sentiment_regime"] = trades_df["sentiment"].astype(str)

    # Performance by regime
    agg_dict = {}
    if pnl_col:
        agg_dict[pnl_col] = ["sum", "mean", "count"]
    if "leverage" in trades_df.columns:
        trades_df["leverage"] = pd.to_numeric(trades_df["leverage"], errors="coerce")
        agg_dict["leverage"] = "mean"

    perf = trades_df.groupby("sentiment_regime", observed=True).agg(agg_dict).round(4)
    if pnl_col and "leverage" in agg_dict:
        perf.columns = ["total_pnl", "avg_pnl_per_trade", "trade_count", "avg_leverage"]
    elif pnl_col:
        perf.columns = ["total_pnl", "avg_pnl_per_trade", "trade_count"]

    print("\n=== PERFORMANCE BY SENTIMENT REGIME ===\n")
    print(perf.to_string())

    if pnl_col:
        trades_df["win"] = (trades_df[pnl_col] > 0).astype(int)
        win_rate = trades_df.groupby("sentiment_regime", observed=True)["win"].mean() * 100
        print("\n=== WIN RATE BY REGIME (%) ===\n")
        print(win_rate.round(2).to_string())

    # Correlation if numeric sentiment
    if trades_df["sentiment"].dtype in [np.float64, np.int64] and pnl_col:
        daily = trades_df.groupby("date_only").agg(total_pnl=(pnl_col, "sum")).reset_index()
        sent_daily = trades_df.groupby("date_only")["sentiment"].mean().reset_index()
        sent_daily.columns = ["date_only", "mean_sentiment"]
        daily = daily.merge(sent_daily, on="date_only", how="left")
        valid = daily[["mean_sentiment", "total_pnl"]].dropna()
        if len(valid) > 2:
            from scipy import stats
            r, p = stats.pearsonr(valid["mean_sentiment"], valid["total_pnl"])
            print(f"\n=== CORRELATION (sentiment vs daily total PnL) ===\n  r = {r:.4f}, p = {p:.4f}")

    if HAS_PLOT and pnl_col and len(perf) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        perf["total_pnl"].plot(kind="bar", ax=axes[0], color="steelblue")
        axes[0].set_title("Total PnL by Sentiment Regime")
        axes[0].set_ylabel("Total PnL")
        axes[0].tick_params(axis="x", rotation=45)
        win_rate.plot(kind="bar", ax=axes[1], color="coral")
        axes[1].set_title("Win Rate (%) by Sentiment Regime")
        axes[1].set_ylabel("Win Rate %")
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        out = Path(__file__).resolve().parent / "output_plots.png"
        plt.savefig(out, dpi=120)
        print(f"\nPlots saved to: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
