import pandas as pd
from pathlib import Path


def fetch_customer_data(data_version="v1"):
    """
    Reads the Telco customer churn dataset for a given version.

    Parameters
    ----------
    data_version : str, optional
        Which version of the dataset to load (e.g., 'v1', 'v2', 'v3').
        Default is 'v1'.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / data_version / "Telco-Customer-Churn.csv"

    if not data_path.is_file():
        raise OSError(f"Unable to locate dataset at:\n{data_path}")

    df = pd.read_csv(data_path)

    print(f"[INFO] Successfully loaded version: {data_version}")
    print(f"[INFO] Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    data = fetch_customer_data("v1")
    print(data.head())