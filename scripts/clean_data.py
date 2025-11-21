from pathlib import Path
import pandas as pd
# import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "online_retail_II.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned.csv"

def clean_data(df):
    df = df.drop_duplicates()
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df = df.dropna(subset=["Customer ID"])

    return df

if __name__ == "__main__":
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_FILE, encoding='unicode_escape')
    df = clean_data(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print("✔ Cleaned file saved at:", OUTPUT_FILE)
