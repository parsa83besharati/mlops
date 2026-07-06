from pathlib import Path

import mlflow
import sys

import mlflow.xgboost
import mlflow.catboost

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Add src to Python path
project_root = Path(__file__).resolve().parent

sys.path.append(str(project_root / "src"))

from data_loader import load_data
from preprocessing import preprocess_data
from features import feature_engineering 

from train import (
    split_data,
    train_all_models
)

from evaluate import (
    evaluate_all_models
)

from mlflow_utils import (
    setup_mlflow,
    log_model_run,
    
)


DATASET_VERSION = "v3"


def main():

    print("=" * 60)
    print("Customer Churn MLOps Pipeline")
    print("=" * 60)

    # ----------------------------------
    # MLflow
    # ----------------------------------

    setup_mlflow()

    # ----------------------------------
    # Load Data
    # ----------------------------------

    print("\nLoading Dataset...")

    df = load_data(DATASET_VERSION)

    # ----------------------------------
    # Preprocessing
    # ----------------------------------

    print("Preprocessing...")

    df = preprocess_data(df)

    # ----------------------------------
    # Feature Engineering
    # ----------------------------------

    print("Feature Engineering...")

    df = feature_engineering(df)

    # ----------------------------------
    # Train / Validation / Test Split
    # ----------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test

    ) = split_data(df)

    # ----------------------------------
    # Train Models
    # ----------------------------------

    print("Training Models...")

    trained_models = train_all_models(
        X_train,
        y_train
    )

    # ----------------------------------
    # Evaluation
    # ----------------------------------

    print("Evaluating Models...")

    evaluation_results = evaluate_all_models(

        trained_models,

        X_validation,
        y_validation,

        X_test,
        y_test

    )

    # ----------------------------------
    # Log to MLflow
    # ----------------------------------

    print("Logging to MLflow...")

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["validation"]["f1"]
    )

    for model_name, info in trained_models.items():

        is_best = (model_name == best_model_name)
        log_model_run(

            model_name=model_name,

            model=info["model"],

            dataset_version=DATASET_VERSION,

            best_params=info["best_params"],

            metrics=evaluation_results[model_name],
            
            X_test=X_test,
            
            y_test=y_test,

            seed=42,

            is_best=is_best
        )


    # ----------------------------------
    # Find Best Model
    # ----------------------------------

    print("\n" + "=" * 60)

    print("Pipeline Finished")

    print("=" * 60)

    print(f"Best Model : {best_model_name}")

    print(
        f"Validation F1 : "
        f"{evaluation_results[best_model_name]['validation']['f1']:.4f}"
    )

    print(
        f"Test F1 : "
        f"{evaluation_results[best_model_name]['test']['f1']:.4f}"
    )


if __name__ == "__main__":

    main()