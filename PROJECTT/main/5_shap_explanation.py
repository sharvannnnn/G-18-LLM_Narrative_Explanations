import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


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


def explain_with_shap():
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

    # Use SHAP KernelExplainer for sklearn model
    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
    explainer = shap.KernelExplainer(ann.predict, background)

    # Explain a few test samples
    test_samples = X_test_scaled[:5]
    shap_values = explainer.shap_values(test_samples)

    # Save summary plot
    shap.summary_plot(shap_values, test_samples, feature_names=X.columns.tolist(), show=False)
    import matplotlib.pyplot as plt
    plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save force plot for first sample (HTML)
    shap.save_html("shap_force_plot.html", shap.force_plot(explainer.expected_value, shap_values[0], test_samples[0], feature_names=X.columns.tolist()))

    print("SHAP explanations saved:")
    print("  shap_summary_plot.png")
    print("  shap_force_plot.html")


if __name__ == "__main__":
    explain_with_shap()