from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "cleaned.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cohort_matrix.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")
df["CohortMonth"] = df.groupby("Customer ID")["InvoiceDate"].transform(lambda x: x.min().to_period("M"))

df["CohortIndex"] = (df["OrderMonth"] - df["CohortMonth"]).apply(lambda x: x.n)

cohort_data = (
    df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
    .nunique()
    .reset_index()
)

cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="Customer ID")

OUTPUT_FILE.parent.mkdir(exist_ok=True)
cohort_pivot.to_csv(OUTPUT_FILE)

print("✔ Cohort matrix saved to", OUTPUT_FILE)
