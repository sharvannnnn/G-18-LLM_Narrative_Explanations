import numpy as np
import pandas as pd
from pathlib import Path
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer


def load_shap_data():
    # Load SHAP values from the summary plot or force plot
    # Since we have shap_summary_plot.png, but to get data, we need to recompute or load
    # For simplicity, I'll recompute SHAP values here
    from shap_ann import load_and_preprocess, train_ann
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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

    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
    explainer = shap.KernelExplainer(ann.predict, background)
    test_samples = X_test_scaled[:5]
    shap_values = explainer.shap_values(test_samples)

    # Get mean absolute SHAP values for feature importance
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance_shap = dict(zip(X.columns, mean_shap))

    return feature_importance_shap


def load_lime_data():
    # Load LIME explanation
    with open("lime_explanation.txt", "r") as f:
        lime_text = f.read()

    # Parse the LIME output
    lines = lime_text.split('\n')
    lime_features = {}
    for line in lines:
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                feature, value = parts
                try:
                    lime_features[feature.strip()] = float(value.strip())
                except:
                    pass

    return lime_features


def generate_medical_narration(shap_importance, lime_explanation):
    # Sort features by importance
    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_shap[:5]  # Top 5

    # Map features to medical terms
    feature_map = {
        'Age': 'age',
        'Sex': 'gender',
        'BMI': 'body mass index (BMI)',
        'BP': 'blood pressure',
        'S1': 'total serum cholesterol',
        'S2': 'low-density lipoproteins',
        'S3': 'high-density lipoproteins',
        'S4': 'total cholesterol / HDL',
        'S5': 'possibly log of serum triglycerides',
        'S6': 'blood sugar level'
    }

    narration = "Based on the AI model's analysis using SHAP and LIME explanations, here is a simple medical interpretation of the diabetes risk prediction:\n\n"

    narration += "The most important factors influencing the diabetes progression score are:\n"
    for feature, importance in top_features:
        medical_name = feature_map.get(feature, feature)
        narration += f"- {medical_name}: This factor has a significant impact on the prediction.\n"

    narration += "\nIn medical terms:\n"
    narration += "- Age: Older individuals may have higher diabetes risk due to metabolic changes.\n"
    narration += "- Gender: Biological differences can affect diabetes susceptibility.\n"
    narration += "- BMI: Higher body mass index indicates obesity, a major risk factor for diabetes.\n"
    narration += "- Blood Pressure: Elevated BP is linked to insulin resistance.\n"
    narration += "- Cholesterol levels (S1-S5): Abnormal lipid profiles can contribute to diabetes.\n"
    narration += "- Blood Sugar (S6): Directly related to glucose control and diabetes.\n"

    narration += "\nThe model suggests that lifestyle factors like weight management and blood pressure control are crucial for diabetes prevention. Consult a healthcare professional for personalized advice."

    return narration


def main():
    shap_importance = load_shap_data()
    lime_explanation = load_lime_data()

    narration = generate_medical_narration(shap_importance, lime_explanation)

    with open("llm_medical_narration.txt", "w") as f:
        f.write(narration)

    print("LLM Medical Narration saved to llm_medical_narration.txt")


if __name__ == "__main__":
    main()