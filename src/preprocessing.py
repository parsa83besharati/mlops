from pathlib import Path
import pandas as pd


def clean_and_encode(input_df):
    """
    Preprocess the Telco Customer Churn dataset by removing unnecessary columns,
    converting data types, handling missing values, and applying one-hot encoding.
    """
    
    # ---------------------------------------------
    # Drop irrelevant features from the dataset
    # ---------------------------------------------
    unwanted_features = [
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
    
    available_features = [f for f in unwanted_features if f in input_df.columns]
    input_df = input_df.drop(columns=available_features)
    
    # ---------------------------------------------
    # Ensure Total Charges is properly formatted
    # ---------------------------------------------
    input_df["Total Charges"] = pd.to_numeric(
        input_df["Total Charges"],
        errors="coerce"
    )
    
    # ---------------------------------------------
    # Drop any rows with missing data
    # ---------------------------------------------
    input_df = input_df.dropna()
    
    # ---------------------------------------------
    # Identify categorical features for encoding
    # ---------------------------------------------
    categorical_features = input_df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    # Exclude target variable from encoding
    if "Churn Value" in categorical_features:
        categorical_features.remove("Churn Value")
    
    # ---------------------------------------------
    # Apply one-hot encoding with drop_first
    # ---------------------------------------------
    input_df = pd.get_dummies(
        input_df,
        columns=categorical_features,
        drop_first=True
    )
    
    # Convert boolean columns to integers (0/1)
    boolean_cols = input_df.select_dtypes(include="bool").columns
    input_df[boolean_cols] = input_df[boolean_cols].astype(int)
    
    return input_df


def store_processed_data(dataframe, version_label="v2"):
    """
    Save the cleaned and encoded dataset to the specified version folder.
    """
    root_dir = Path(__file__).resolve().parent.parent
    output_path = root_dir / "data" / version_label / "Telco-Customer-Churn.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Processed data saved at:")
    print(f"{output_path}")


if __name__ == "__main__":
    
    root_dir = Path(__file__).resolve().parent.parent
    raw_data_path = root_dir / "data" / "v1" / "Telco-Customer-Churn.csv"
    
    raw_df = pd.read_csv(raw_data_path)
    print(f"Original dataset shape: {raw_df.shape}")
    
    processed_df = clean_and_encode(raw_df)
    print(f"Processed dataset shape: {processed_df.shape}")
    
    store_processed_data(processed_df)
    
    print("\nPreview of processed data:")
    print(processed_df.head())