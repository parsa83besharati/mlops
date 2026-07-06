from pathlib import Path
import pandas as pd


def load_data(version="v1"):
    """
    Load dataset based on version.

    Parameters
    ----------
    version : str
        Dataset version (v1, v2, v3)

    Returns
    -------
    pandas.DataFrame
    """

    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / version / "Telco-Customer-Churn.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{file_path}")

    df = pd.read_csv(file_path)

    print(f"Loaded {version} dataset")
    print(f"Shape: {df.shape}")

    return df

if __name__ == "__main__":

    df = load_data("v1")

    print(df.head())