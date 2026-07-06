from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_engineering(df):
    """
    Perform feature engineering and normalization.
    """

    df = df.copy()

    # ==========================================================
    # Feature 1: Average Monthly Charge
    # ==========================================================

    df["Avg Monthly Charge"] = np.where(
        df["Tenure Months"] == 0,
        0,
        df["Total Charges"] / df["Tenure Months"]
    )

    # ==========================================================
    # Feature 2: Is New Customer
    # ==========================================================

    df["Is New Customer"] = (
        df["Tenure Months"] < 12
    ).astype(int)

    # ==========================================================
    # Feature 3: High Monthly Charge
    # ==========================================================

    average_monthly_charge = df["Monthly Charges"].mean()

    df["High Monthly Charge"] = (
        df["Monthly Charges"] > average_monthly_charge
    ).astype(int)

    # ==========================================================
    # Normalization
    # ==========================================================

    scaler = StandardScaler()

    numerical_columns = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Avg Monthly Charge"
    ]

    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

    return df


def save_feature_data(df, version="v3"):
    """
    Save engineered dataset.
    """

    project_root = Path(__file__).resolve().parent.parent

    save_path = (
        project_root
        / "data"
        / version
        / "Telco-Customer-Churn.csv"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, index=False)

    print("\nDataset saved successfully:")
    print(save_path)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "data"
        / "v2"
        / "Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(input_path)

    print("Input Shape:", df.shape)

    feature_df = feature_engineering(df)

    print("Output Shape:", feature_df.shape)

    save_feature_data(feature_df)

    print("\nFirst 5 rows:\n")
    print(feature_df.head())

    print("\nNew Features Added:\n")

    print("- Avg Monthly Charge")
    print("- Is New Customer")
    print("- High Monthly Charge")

    print("\nDataset Information")
    print("-" * 40)
    print(f"Rows: {feature_df.shape[0]}")
    print(f"Columns: {feature_df.shape[1]}")

    print("\nTarget Distribution:\n")
    print(feature_df["Churn Value"].value_counts())