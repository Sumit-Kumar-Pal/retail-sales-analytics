from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "cleaned.csv"
CHARTS_DIR = BASE_DIR / "charts"
OUTPUT_FILE = CHARTS_DIR / "forecast.png"

df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly = df.groupby("Month")["TotalPrice"].sum()

# Use simple moving average with NumPy
window = 3
forecast = np.convolve(monthly, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(monthly.index.astype(str), monthly.values, label="Actual")
plt.plot(monthly.index[window-1:].astype(str), forecast, label="Forecast (SMA)")
plt.xticks(rotation=45)
plt.legend()
plt.title("Sales Forecast")
plt.tight_layout()

CHARTS_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_FILE)
