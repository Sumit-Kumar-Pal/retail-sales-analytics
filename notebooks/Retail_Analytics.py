from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"

INPUT_FILE = DATA_DIR / "online_retail_II.csv"

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset.")
df = pd.read_csv(INPUT_FILE, encoding="unicode_escape")

print("Dataset Loaded.")
print(df.head())
print(df.info())

# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\nCleaning Data.")

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df = df.dropna(subset=["Customer ID"])

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["Price"]

print("Data cleaned.")
print(df.head())

# ============================================================
# 4. SALES OVERVIEW
# ============================================================

print("\nGenerating Monthly Sales Analysis.")

df["Month"] = df["InvoiceDate"].dt.to_period("M")
monthly_sales = df.groupby("Month")["TotalPrice"].sum()

CHARTS_DIR.mkdir(exist_ok=True)

plt.figure(figsize=(10, 5))
monthly_sales.plot()
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "monthly_sales.png")
plt.close()

print("monthly_sales.png saved.")

# ============================================================
# 5. PRODUCT PERFORMANCE
# ============================================================

print("\nFinding Top 10 Products.")

top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
top_products.plot(kind="bar")
plt.title("Top 10 Bestselling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "top_products.png")
plt.close()

print("top_products.png saved.")

# ============================================================
# 6. RFM ANALYSIS
# ============================================================

print("\nRunning RFM Analysis.")

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "Invoice": "count",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Rank RFM values
rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
rfm["F"] = pd.qcut(rfm["Frequency"], 4, labels=[1, 2, 3, 4])
rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4])

rfm["RFM_Score"] = rfm[["R", "F", "M"]].sum(axis=1)

plt.figure(figsize=(8, 5))
plt.hist(rfm["RFM_Score"], bins=10)
plt.title("RFM Score Distribution")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "rfm_segments.png")
plt.close()

rfm.to_csv(DATA_DIR / "rfm_scores.csv")

print("RFM analysis completed. Saved rfm_scores.csv.")

# ============================================================
# 7. COHORT ANALYSIS
# ============================================================

print("\nRunning Cohort Analysis.")

df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")

# First purchase month per customer
df["CohortMonth"] = (
    df.groupby("Customer ID")["InvoiceDate"]
    .transform(lambda x: x.min().to_period("M"))
)

df["CohortIndex"] = (df["OrderMonth"] - df["CohortMonth"]).apply(lambda x: x.n)

cohort_data = (
    df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
    .nunique()
    .reset_index()
)

cohort_matrix = cohort_data.pivot(
    index="CohortMonth",
    columns="CohortIndex",
    values="Customer ID"
)

cohort_matrix.to_csv(DATA_DIR / "cohort_matrix.csv")

print("Cohort matrix saved.")

# ============================================================
# 8. SALES FORECASTING (Simple Moving Average)
# ============================================================

print("\nForecasting Sales.")

monthly_values = monthly_sales.reset_index()
monthly_values["Month"] = monthly_values["Month"].astype(str)

window = 3
forecast = np.convolve(
    monthly_values["TotalPrice"],
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))
plt.plot(monthly_values["Month"], monthly_values["TotalPrice"], label="Actual")
plt.plot(monthly_values["Month"][window-1:], forecast, label="Forecast SMA(3)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "forecast.png")
plt.close()

print("forecast.png saved.")

# ============================================================
# DONE
# ============================================================

print("\nRetail Analytics Notebook executed successfully.")
print("Charts saved in: charts/")
print("Processed data saved in: data/")
