
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = Path("Ddiabetes_data.csv")
if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path.resolve()}")

df = pd.read_csv(file_path)

# Normalize column names
df.columns = df.columns.str.strip()

print("Dataset columns:", list(df.columns))
print("\nFirst 10 rows:")
print(df.head(10))
print("\nMissing values:")
print(df.isnull().sum())

# Identify target column
target_column = None
for candidate in ["Outcome", "Target", "target", "outcome"]:
    if candidate in df.columns:
        target_column = candidate
        break
if target_column is None:
    raise ValueError("Target column not found. Expected one of: Outcome, Target")

# Replace zeros with NaN for numeric columns where zero is invalid
zero_invalid_candidates = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_invalid_candidates += [c.lower() for c in zero_invalid_candidates]
zero_invalid_columns = [col for col in df.columns if col in zero_invalid_candidates]
if zero_invalid_columns:
    df[zero_invalid_columns] = df[zero_invalid_columns].replace(0, pd.NA)
    df[zero_invalid_columns] = df[zero_invalid_columns].apply(pd.to_numeric, errors="coerce")

# Ensure numeric columns are numeric and fill missing values with median
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
if numeric_columns:
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_medians = df[numeric_columns].median()
    df[numeric_columns] = df[numeric_columns].fillna(numeric_medians)

# Prepare features and target
X = df.drop(columns=[target_column])
y = df[target_column]

if X.empty:
    raise ValueError("No feature columns available after dropping target column.")

# Split data
stratify_target = None
if y.nunique() > 1:
    class_counts = y.value_counts()
    if class_counts.min() >= 2 and y.dtype.kind in "biu":
        stratify_target = y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_target,
)

# Standard scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing complete!")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")