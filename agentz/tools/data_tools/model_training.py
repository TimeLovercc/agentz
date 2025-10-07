"""Model training tool for training machine learning models."""

from typing import Union, Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from agents import function_tool


@function_tool
async def train_model(
    file_path: str,
    target_column: str,
    model_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42
) -> Union[Dict[str, Any], str]:
    """Trains machine learning models on a dataset.

    Args:
        file_path: Path to the dataset file
        target_column: Name of the target column to predict
        model_type: Type of model to train. Options:
            - "auto": Automatically detect and use best model
            - "random_forest": Random Forest
            - "logistic_regression": Logistic Regression (classification)
            - "linear_regression": Linear Regression (regression)
            - "decision_tree": Decision Tree
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing:
            - model_type: Type of model trained
            - problem_type: "classification" or "regression"
            - train_score: Training score
            - test_score: Testing score
            - cross_val_scores: Cross-validation scores (mean and std)
            - feature_importance: Feature importance scores (if available)
            - predictions_sample: Sample of predictions vs actual values
        Or error message string if training fails
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

        # Handle categorical features in X
        X = pd.get_dummies(X, drop_first=True)

        # Determine problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20

        # Encode target if categorical
        if is_classification and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Select model
        if model_type == "auto":
            if is_classification:
                model = RandomForestClassifier(random_state=random_state, n_estimators=100)
                model_name = "Random Forest Classifier"
            else:
                model = RandomForestRegressor(random_state=random_state, n_estimators=100)
                model_name = "Random Forest Regressor"
        elif model_type == "random_forest":
            if is_classification:
                model = RandomForestClassifier(random_state=random_state, n_estimators=100)
                model_name = "Random Forest Classifier"
            else:
                model = RandomForestRegressor(random_state=random_state, n_estimators=100)
                model_name = "Random Forest Regressor"
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model_name = "Logistic Regression"
        elif model_type == "linear_regression":
            model = LinearRegression()
            model_name = "Linear Regression"
        elif model_type == "decision_tree":
            if is_classification:
                model = DecisionTreeClassifier(random_state=random_state)
                model_name = "Decision Tree Classifier"
            else:
                model = DecisionTreeRegressor(random_state=random_state)
                model_name = "Decision Tree Regressor"
        else:
            return f"Unknown model type: {model_type}"

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        if is_classification:
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            metric_name = "accuracy"
        else:
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            metric_name = "r2_score"

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)

        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(X.columns, model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        # Sample predictions
        predictions_sample = []
        for i in range(min(10, len(y_test))):
            predictions_sample.append({
                "actual": float(y_test.iloc[i]) if hasattr(y_test, 'iloc') else float(y_test[i]),
                "predicted": float(test_pred[i]),
            })

        result = {
            "model_type": model_name,
            "problem_type": "classification" if is_classification else "regression",
            "train_score": float(train_score),
            "test_score": float(test_score),
            "metric": metric_name,
            "cross_val_mean": float(cv_scores.mean()),
            "cross_val_std": float(cv_scores.std()),
            "feature_importance": {k: float(v) for k, v in list(feature_importance.items())[:10]},  # Top 10
            "predictions_sample": predictions_sample,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return result

    except Exception as e:
        return f"Error training model: {str(e)}"
