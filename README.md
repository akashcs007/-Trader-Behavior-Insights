# -Trader-Behavior-Insights

## Overview
This repository contains an exploratory data analysis project investigating the relationship between trader performance and broader market sentiment[cite: 11, 12]. [cite_start]The objective is to uncover hidden patterns in trader behavior across different market regimes and extract actionable insights to drive smarter trading strategies.



## Key Insights & Findings
Our analysis reveals a significant disconnect between a trader's win rate and their actual bottom-line profitability across different market regimes:

* **The "Fear" Premium:** The data demonstrates that the **"Fear"** sentiment regime generates the vast majority of Total PnL, despite exhibiting a relatively modest win rate of approximately 41%. This suggests that fearful market conditions offer highly asymmetric returns, where successful trades yield massive profits.
* **The "Extreme Greed" Illusion:** Conversely, the **"Extreme Greed"** regime presents the highest overall win rate (approaching 50%) but generates almost negligible Total PnL. This indicates a dynamic where traders win frequently, but the profit per trade is minimal, or losses from the remaining trades are disproportionately large.
* **Strategic Implication:** Trading systems should dynamically adjust risk and capital allocation based on sentiment. Sizing up positions during "Fear" regimes maximizes exposure to high-yield opportunities, while tightening risk management during "Extreme Greed" protects capital from low-value, high-frequency churn.

## Data Sources
The analysis integrates two primary datasets:
1. **Bitcoin Market Sentiment Dataset:** Daily classifications of market sentiment (e.g., Fear, Greed, Extreme Greed, Neutral)[cite: 6, 7].
2. **Historical Trader Data (Hyperliquid):** Granular trade execution data including account details, execution price, side, size, and closed PnL[cite: 8, 9, 10].

## Script Features (`run_analysis.py`)
The pipeline is designed to be robust and handle various data formats:
* **Flexible Data Loading:** Automatically detects and loads `.csv`, `.parquet`, or `.xlsx` files.
* **Timestamp Handling:** Standardizes Unix timestamps (milliseconds or seconds) and string-based dates into a unified `date_only` format for accurate merging.
* **Automated Categorization:** Bins numeric sentiment scores into distinct categorical regimes (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).
* **Statistical Analysis:** Calculates Pearson correlation (`r` and `p-value`) between daily mean sentiment and daily total PnL.
* **Automated Visualization:** Generates and saves a two-panel bar chart (`output_plots.png`) comparing Total PnL and Win Rate side-by-side.

## Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. You will need the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
