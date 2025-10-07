"""Model evaluation tool for assessing model performance."""

from typing import Union, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from agents import function_tool


@function_tool
async def evaluate_model(
    file_path: str,
    target_column: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42
) -> Union[Dict[str, Any], str]:
    """Evaluates machine learning model performance with comprehensive metrics.

    Args:
        file_path: Path to the dataset file
        target_column: Name of the target column to predict
        model_type: Type of model to evaluate (random_forest, decision_tree, etc.)
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing:
            - problem_type: "classification" or "regression"
            - metrics: Performance metrics
            - confusion_matrix: Confusion matrix (for classification)
            - classification_report: Detailed classification report
            - cross_validation: Cross-validation results
            - error_analysis: Error distribution analysis
        Or error message string if evaluation fails
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            return f"File not found: {file_path}"

        # Load dataset
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            return f"Unsupported file format: {file_path.suffix}"

        if target_column not in df.columns:
            return f"Target column '{target_column}' not found in dataset"

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Determine problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20

        # Encode target if categorical
        original_labels = None
        if is_classification and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            original_labels = le.classes_
            y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        if is_classification:
            model = RandomForestClassifier(random_state=random_state, n_estimators=100)
        else:
            model = RandomForestRegressor(random_state=random_state, n_estimators=100)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result = {
            "problem_type": "classification" if is_classification else "regression",
        }

        if is_classification:
            # Classification metrics
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            }
            result["metrics"] = metrics

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            result["confusion_matrix"] = cm.tolist()

            # Classification report
            if original_labels is not None:
                target_names = [str(label) for label in original_labels]
            else:
                target_names = [str(i) for i in sorted(np.unique(y))]

            class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
            result["classification_report"] = class_report

            # Per-class accuracy
            per_class_accuracy = {}
            for i, label in enumerate(target_names):
                mask = y_test == i
                if mask.sum() > 0:
                    per_class_accuracy[label] = float(accuracy_score(y_test[mask], y_pred[mask]))
            result["per_class_accuracy"] = per_class_accuracy

        else:
            # Regression metrics
            metrics = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
                "root_mean_squared_error": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
                "mean_absolute_percentage_error": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
            }
            result["metrics"] = metrics

            # Error analysis
            errors = y_test - y_pred
            error_analysis = {
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "min_error": float(np.min(errors)),
                "max_error": float(np.max(errors)),
                "median_error": float(np.median(errors)),
            }
            result["error_analysis"] = error_analysis

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        result["cross_validation"] = {
            "scores": cv_scores.tolist(),
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        }

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(X.columns, model.feature_importances_))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            result["feature_importance"] = {k: float(v) for k, v in list(sorted_importance.items())[:10]}

        return result

    except Exception as e:
        return f"Error evaluating model: {str(e)}"
