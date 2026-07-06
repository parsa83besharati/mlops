import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix

EXPERIMENT_NAME = "Customer Churn Prediction"


def setup_mlflow():
    """
    Create (or load) MLflow experiment.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_confusion_matrix_values(model, X_test, y_test):
    """
    Log confusion matrix values as metrics (no image saved).
    """
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    mlflow.log_metric("confusion_tn", tn)
    mlflow.log_metric("confusion_fp", fp)
    mlflow.log_metric("confusion_fn", fn)
    mlflow.log_metric("confusion_tp", tp)


def log_model_run(
    model_name,
    model,
    dataset_version,
    best_params,
    metrics,
    X_test,
    y_test,
    seed=42,
    is_best=False
):
    """
    Log one trained model into MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        # ==========================
        # Parameters
        # ==========================

        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("seed", seed)
        mlflow.log_param("best_model", is_best)

        for key, value in best_params.items():
            mlflow.log_param(key, value)

        # ==========================
        # Validation Metrics
        # ==========================

        validation = metrics["validation"]

        mlflow.log_metric("val_accuracy", validation["accuracy"])
        mlflow.log_metric("val_precision", validation["precision"])
        mlflow.log_metric("val_recall", validation["recall"])
        mlflow.log_metric("val_f1", validation["f1"])
        mlflow.log_metric("val_roc_auc", validation["roc_auc"])

        # ==========================
        # Test Metrics
        # ==========================

        test = metrics["test"]

        mlflow.log_metric("test_accuracy", test["accuracy"])
        mlflow.log_metric("test_precision", test["precision"])
        mlflow.log_metric("test_recall", test["recall"])
        mlflow.log_metric("test_f1", test["f1"])
        mlflow.log_metric("test_roc_auc", test["roc_auc"])

        # ==========================
        # Cross Validation
        # ==========================

        mlflow.log_metric("cv_f1", metrics["cv_score"])

        # ==========================
        # Save Model
        # ==========================

        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model"
            )
        elif isinstance(model, CatBoostClassifier):
            mlflow.catboost.log_model(
                cb_model=model,
                artifact_path="model"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

        # ==========================
        # Log Confusion Matrix Values (no image)
        # ==========================

        log_confusion_matrix_values(model, X_test, y_test)