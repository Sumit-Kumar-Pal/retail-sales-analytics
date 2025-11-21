from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned.csv"
CHARTS_DIR = PROJECT_ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(10,5))
monthly.plot()
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "monthly_sales.png")

top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")
plt.title("Top 10 Products")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "top_products.png")
