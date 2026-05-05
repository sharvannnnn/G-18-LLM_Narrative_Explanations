import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from captum.attr import visualization as viz


class ANNRegressor(nn.Module):
    def __init__(self, input_size):
        super(ANNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


def train_pytorch_ann(X_train, y_train, input_size):
    model = ANNRegressor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model


def explain_with_captum():
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

    model = train_pytorch_ann(X_train_scaled, y_train, X_train_scaled.shape[1])

    # Convert test sample to tensor
    test_sample = torch.tensor(X_test_scaled[0:1], dtype=torch.float32)

    # Integrated Gradients
    ig = IntegratedGradients(model)
    attr_ig = ig.attribute(test_sample, target=0)

    # Layer Conductance for first hidden layer
    layer_cond = LayerConductance(model, model.fc1)
    attr_lc = layer_cond.attribute(test_sample, target=0)

    # Neuron Conductance for first neuron in first layer
    neuron_cond = NeuronConductance(model, model.fc1)
    attr_nc = neuron_cond.attribute(test_sample, neuron_selector=0)

    # Save attributions
    with open("captum_attributions.txt", "w") as f:
        f.write("Captum Explanations for first test sample:\n")
        f.write(f"Integrated Gradients: {attr_ig.detach().numpy()}\n")
        f.write(f"Layer Conductance (fc1): {attr_lc.detach().numpy()}\n")
        f.write(f"Neuron Conductance (fc1, neuron 0): {attr_nc.detach().numpy()}\n")

    print("Captum explanations saved:")
    print("  captum_attributions.txt")


if __name__ == "__main__":
    explain_with_captum()