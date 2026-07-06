from pathlib import Path
import pandas as pd


def preprocess_data(df):
    """
    Preprocess IBM Telco Customer Churn dataset.
    """

    # -----------------------------
    # Remove unnecessary columns
    # -----------------------------
    columns_to_drop = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "CLTV",
    "Churn Label",
    "Churn Score",
    "Churn Category",
    "Churn Reason"
    ]

    existing_columns = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_columns)

    # Convert Total Charges to numeric
    df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce")

    # -----------------------------
    # Remove missing values
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # Find categorical columns
    # -----------------------------
    categorical_columns = (
    df.select_dtypes(include=["object", "string"])
      .columns
      .tolist())

    # Remove target column if needed
    if "Churn Value" in categorical_columns:
        categorical_columns.remove("Churn Value")

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    bool_columns = df.select_dtypes(include="bool").columns
    df[bool_columns] = df[bool_columns].astype(int)


    return df


def save_processed_data(df, version="v2"):
    """
    Save processed dataset.
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

    print(f"\nDataset saved successfully:")
    print(save_path)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    raw_path = (
        project_root
        / "data"
        / "v1"
        / "Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(raw_path)

    print("Original Shape:", df.shape)

    processed_df = preprocess_data(df)

    print("Processed Shape:", processed_df.shape)

    save_processed_data(processed_df)

    print("\nFirst 5 rows:")
    print(processed_df.head())