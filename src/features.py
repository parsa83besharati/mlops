from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_engineered_features(dataset):
    """
    Create new features and apply normalization to the dataset.
    """
    
    df = dataset.copy()
    
    # ----------------------------------------------------------
    # Feature 1: Average monthly spend over tenure period
    # ----------------------------------------------------------
    df["Avg Monthly Charge"] = np.where(
        df["Tenure Months"] == 0,
        0,
        df["Total Charges"] / df["Tenure Months"]
    )
    
    # ----------------------------------------------------------
    # Feature 2: Flag for recent customers (less than 1 year)
    # ----------------------------------------------------------
    df["Is New Customer"] = (df["Tenure Months"] < 12).astype(int)
    
    # ----------------------------------------------------------
    # Feature 3: Indicator for above-average monthly charges
    # ----------------------------------------------------------
    avg_monthly = df["Monthly Charges"].mean()
    df["High Monthly Charge"] = (df["Monthly Charges"] > avg_monthly).astype(int)
    
    # ----------------------------------------------------------
    # Standardize numerical features
    # ----------------------------------------------------------
    scaler = StandardScaler()
    
    numeric_features = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Avg Monthly Charge"
    ]
    
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df


def export_featured_data(dataframe, dataset_version="v3"):
    """
    Save the feature-engineered dataset to the specified version folder.
    """
    project_root = Path(__file__).resolve().parent.parent
    output_file = project_root / "data" / dataset_version / "Telco-Customer-Churn.csv"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False)
    
    print(f"\n[INFO] Feature-engineered data exported to:")
    print(f"{output_file}")


if __name__ == "__main__":
    
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "data" / "v2" / "Telco-Customer-Churn.csv"
    
    raw_data = pd.read_csv(input_file)
    print(f"Input shape: {raw_data.shape}")
    
    engineered_data = create_engineered_features(raw_data)
    print(f"Output shape: {engineered_data.shape}")
    
    export_featured_data(engineered_data)
    
    print("\nPreview of engineered data:")
    print(engineered_data.head())
    
    print("\nNew features created:")
    print("  • Avg Monthly Charge")
    print("  • Is New Customer")
    print("  • High Monthly Charge")
    
    print("\n" + "=" * 40)
    print("Dataset Summary")
    print("=" * 40)
    print(f"Total rows: {engineered_data.shape[0]}")
    print(f"Total columns: {engineered_data.shape[1]}")
    
    print("\nTarget variable distribution (Churn Value):")
    print(engineered_data["Churn Value"].value_counts())