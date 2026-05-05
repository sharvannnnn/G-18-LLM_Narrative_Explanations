import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def train_and_save_results():
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

    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=1000,
    )
    ann.fit(X_train_scaled, y_train)

    y_train_pred = ann.predict(X_train_scaled)
    y_test_pred = ann.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    result_path = Path("ann_results.txt")
    with result_path.open("w") as f:
        f.write("ANN Regression Results\n")
        f.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}\n")
        f.write("\nTrain metrics:\n")
        f.write(f"  RMSE: {train_rmse:.4f}\n")
        f.write(f"  MAE:  {train_mae:.4f}\n")
        f.write(f"  R2:   {train_r2:.4f}\n")
        f.write("\nTest metrics:\n")
        f.write(f"  RMSE: {test_rmse:.4f}\n")
        f.write(f"  MAE:  {test_mae:.4f}\n")
        f.write(f"  R2:   {test_r2:.4f}\n")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=16)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title("ANN: Actual vs Predicted on Test Set")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ann_actual_vs_predicted.png", dpi=150)
    plt.close()

    if hasattr(ann, "loss_curve_") and len(ann.loss_curve_) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(ann.loss_curve_, marker="o", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("ANN Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ann_loss_curve.png", dpi=150)
        plt.close()

    print("ANN results saved:")
    print(f"  {result_path}")
    print("  ann_actual_vs_predicted.png")
    if hasattr(ann, "loss_curve_") and len(ann.loss_curve_) > 0:
        print("  ann_loss_curve.png")


if __name__ == "__main__":
    train_and_save_results()
