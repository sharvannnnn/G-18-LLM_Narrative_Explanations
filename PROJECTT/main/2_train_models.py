import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    target_column = next(
        (col for col in ["Outcome", "Target", "outcome", "target"] if col in df.columns),
        None,
    )
    if target_column is None:
        raise ValueError("Could not find a target column. Expected Outcome or Target.")

    zero_invalid_names = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_invalid_names += [name.lower() for name in zero_invalid_names]
    zero_invalid_columns = [col for col in df.columns if col in zero_invalid_names]

    if zero_invalid_columns:
        df[zero_invalid_columns] = df[zero_invalid_columns].replace(0, pd.NA)

    numeric_columns = [
        col
        for col in df.columns
        if np.issubdtype(df[col].dtype, np.number) or col in zero_invalid_columns
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    medians = df[numeric_columns].median()
    df[numeric_columns] = df[numeric_columns].fillna(medians)

    X = df.drop(columns=[target_column])
    y_regression = df[target_column].astype(float)
    y_classification = (y_regression > y_regression.median()).astype(int)

    return X, y_regression, y_classification


def split_and_scale(X, y, stratify=None, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_ann(X_train, y_train):
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=1000,
    )
    ann.fit(X_train, y_train)
    return ann


def evaluate_regression(model, X, y, name="Model"):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\n{name} regression results")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    return y_pred


def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classification(model, X, y, name="Model"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n{name} classification results")
    print(f"  Accuracy: {accuracy:.4f}")
    print(classification_report(y, y_pred, digits=4))
    return y_pred


def main():
    data_path = Path("Ddiabetes_data.csv")
    X, y_regression, y_classification = load_and_preprocess(data_path)

    print("Loaded and preprocessed data")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Regression target median threshold: {y_regression.median():.4f}")

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_and_scale(
        X,
        y_regression,
        stratify=None,
    )
    ann_model = train_ann(X_train_reg, y_train_reg)
    evaluate_regression(ann_model, X_train_reg, y_train_reg, name="ANN (train)")
    evaluate_regression(ann_model, X_test_reg, y_test_reg, name="ANN (test)")

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_and_scale(
        X,
        y_classification,
        stratify=y_classification,
    )
    logreg_model = train_logistic_regression(X_train_clf, y_train_clf)
    evaluate_classification(logreg_model, X_train_clf, y_train_clf, name="Logistic Regression (train)")
    evaluate_classification(logreg_model, X_test_clf, y_test_clf, name="Logistic Regression (test)")


if __name__ == "__main__":
    main()
