import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer


def load_and_preprocess(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

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
    y = df[target_column].astype(float)
    return X, y


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


def explain_with_lime():
    data_path = Path("Ddiabetes_data.csv")
    X, y = load_and_preprocess(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ann = train_ann(X_train_scaled, y_train)

    # LIME explainer
    explainer = LimeTabularExplainer(
        X_train_scaled,
        feature_names=X.columns.tolist(),
        class_names=["Target"],
        mode="regression"
    )

    # Explain first test sample
    exp = explainer.explain_instance(X_test_scaled[0], ann.predict, num_features=10)

    # Save explanation as HTML
    exp.save_to_file("lime_explanation.html")

    # Also save as text
    with open("lime_explanation.txt", "w") as f:
        f.write("LIME Explanation for first test sample:\n")
        f.write(str(exp.as_list()))

    print("LIME explanations saved:")
    print("  lime_explanation.html")
    print("  lime_explanation.txt")


if __name__ == "__main__":
    explain_with_lime()