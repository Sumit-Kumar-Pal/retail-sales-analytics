from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "cleaned.csv"
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
OUTPUT_RFM = DATA_DIR / "rfm.csv"
OUTPUT_CHART = CHARTS_DIR / "rfm_segments.png"

df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Convert to scores (1–4)
rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
rfm["F"] = pd.qcut(rfm["Frequency"], 4, labels=[1,2,3,4])
rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])

rfm["RFM_Score"] = rfm[["R","F","M"]].sum(axis=1)

plt.hist(rfm["RFM_Score"], bins=10)
plt.title("RFM Score Distribution")

CHARTS_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_CHART)

rfm.to_csv(OUTPUT_RFM)
print("RFM analysis saved to", OUTPUT_RFM)
