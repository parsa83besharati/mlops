from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X, y):
    """
    Evaluate one trained model.
    """

    y_pred = model.predict(X)

    # Some models may not support predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    results = {

        "accuracy": accuracy_score(y, y_pred),

        "precision": precision_score(y, y_pred),

        "recall": recall_score(y, y_pred),

        "f1": f1_score(y, y_pred),

        "roc_auc": roc_auc_score(y, y_prob)
        if y_prob is not None else None,

        "confusion_matrix": confusion_matrix(y, y_pred)

    }

    return results


def evaluate_all_models(trained_models,
                        X_validation,
                        y_validation,
                        X_test,
                        y_test):
    """
    Evaluate all trained models on Validation and Test sets.
    """

    evaluation_results = {}

    for name, info in trained_models.items():

        print("=" * 60)
        print(f"Evaluating {name}")
        print("=" * 60)

        model = info["model"]

        validation_metrics = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        evaluation_results[name] = {

            "validation": validation_metrics,

            "test": test_metrics,

            "best_params": info["best_params"],

            "cv_score": info["cv_score"]

        }

        print("\nValidation Metrics")
        
        for key, value in validation_metrics.items():
            print(f"{key}: {value}")

        print("\nTest Metrics")
        
        for key, value in test_metrics.items():
            print(f"{key}: {value}")

        print()

    return evaluation_results