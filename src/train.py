from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from evaluate import evaluate_all_models

RANDOM_STATE = 42


def read_preprocessed_data():
    """
    Load the feature-engineered dataset from version 3.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "v3" / "Telco-Customer-Churn.csv"
    return pd.read_csv(data_file)


def prepare_train_val_test_sets(dataframe):
    """
    Split data into training, validation, and test sets (70/15/15 split).
    """
    X = dataframe.drop(columns=["Churn Value"])
    y = dataframe["Churn Value"]
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Second split: 50% of temp = 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def define_model_configs():
    """
    Define all models and their hyperparameter search spaces.
    """
    return {
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


def perform_hyperparameter_tuning(model, param_grid, X_train, y_train):
    """
    Perform grid search with 5-fold stratified cross-validation.
    """
    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search


def train_all_models(X_train, y_train):
    """
    Train and tune all defined models, returning best configurations.
    """
    model_configs = define_model_configs()
    results = {}
    
    for model_name, config in model_configs.items():
        print("=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)
        
        tuned_model = perform_hyperparameter_tuning(
            config["model"],
            config["params"],
            X_train,
            y_train
        )
        
        results[model_name] = {
            "model": tuned_model.best_estimator_,
            "best_params": tuned_model.best_params_,
            "cv_score": tuned_model.best_score_
        }
        
        print(f"Best parameters: {tuned_model.best_params_}")
        print(f"Best CV F1-score: {round(tuned_model.best_score_, 4)}\n")
    
    return results


if __name__ == "__main__":
    
    # Load and prepare data
    dataset = read_preprocessed_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_sets(dataset)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}\n")
    print("Target distribution in training set:")
    print(y_train.value_counts())
    
    # Train all models
    trained_models = train_all_models(X_train, y_train)

    evaluation_results = evaluate_all_models(
        trained_models,
        X_val,
        y_val,
        X_test,
        y_test
    )
    
    # Display final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED - SUMMARY")
    print("=" * 60)
    
    for model_name, result in trained_models.items():
        print("-" * 50)
        print(f"Model: {model_name}")
        print(f"Best params: {result['best_params']}")
        print(f"Best CV F1: {round(result['cv_score'], 4)}")