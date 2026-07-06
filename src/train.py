from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from evaluate import evaluate_all_models

from mlflow_utils import (
    setup_mlflow,
    log_model_run
)

RANDOM_STATE = 42


def load_dataset():

    project_root = Path(__file__).resolve().parent.parent

    file_path = (
        project_root
        / "data"
        / "v3"
        / "Telco-Customer-Churn.csv"
    )

    return pd.read_csv(file_path)


def split_data(df):

    X = df.drop(columns=["Churn Value"])

    y = df["Churn Value"]

    # Train + Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Validation + Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def get_models():
    """
    Return all models with hyperparameter grids.
    """

    models = {

        "Logistic Regression": {
            "model": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),
            "params": {
                "C": [0.1, 1, 10]
            }
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        },

        "XGBoost": {
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        },

        "CatBoost": {
            "model": CatBoostClassifier(
                random_state=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False
            ),
            "params": {
                "depth": [4, 6],
                "learning_rate": [0.01, 0.1]
            }
        }

    }

    return models


def train_model(model, params, X_train, y_train):
    """
    Train one model using GridSearchCV.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=params,

        cv=cv,

        scoring="f1",

        n_jobs=-1,

        refit=True,

        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def train_all_models(X_train, y_train):
    """
    Train all models.
    """

    models = get_models()

    trained_models = {}

    for name, config in models.items():

        print("=" * 60)
        print(f"Training {name}")
        print("=" * 60)

        search = train_model(
            config["model"],
            config["params"],
            X_train,
            y_train
        )

        trained_models[name] = {

            "model": search.best_estimator_,

            "best_params": search.best_params_,

            "cv_score": search.best_score_

        }

        print("Best Parameters:")
        print(search.best_params_)

        print("Best CV Score:")
        print(round(search.best_score_, 4))

        print()

    return trained_models


if __name__ == "__main__":

    setup_mlflow()
    
    df = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(df)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print()

    print(y_train.value_counts())

    project_root = Path(__file__).resolve().parent.parent

    file_path = (
        project_root
        / "data"
        / "v3"
        / "Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(file_path)

    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    trained_models = train_all_models(X_train, y_train)

    evaluation_results = evaluate_all_models(
        trained_models,
        X_val,
        y_val,
        X_test,
        y_test
    )


    for model_name, info in trained_models.items():
        log_model_run(
            model_name=model_name,
            model=info["model"],
            dataset_version="v3",
            best_params=info["best_params"],
            metrics=evaluation_results[model_name],
            seed=42
        )

    print("\nTraining Finished!\n")

    for name, result in trained_models.items():
        print("=" * 50)
        print(name)
        print("Best Parameters:", result["best_params"])