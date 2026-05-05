# Developing a Multi-View XAI Tool with LLM-Driven Narrative Explanations

## Overview
This project demonstrates a complete machine learning workflow for diabetes prediction using a dataset of diabetes-related features. We build and compare two models: an Artificial Neural Network (ANN) for regression and Logistic Regression for classification. To ensure transparency and interpretability, we apply three Explainable AI (XAI) techniques: SHAP, LIME, and Captum. Finally, we generate a simple medical narration based on the explanations.

The goal is to predict diabetes progression (regression) or risk classification (binary), while providing clear, medical-friendly explanations of model decisions.

## Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, shap, lime, torch, captum
- Dataset: `Ddiabetes_data.csv` (diabetes dataset with features like age, BMI, blood pressure, etc.)

## Step-by-Step Process

### 1. Data Preprocessing (`1_data_preprocessing.py`)
**Purpose**: Load the raw diabetes dataset, clean it, and prepare it for modeling.

**What it does**:
- Loads `Ddiabetes_data.csv`
- Handles missing values by replacing zeros with NaN for invalid features (e.g., BMI can't be 0)
- Converts data types and fills missing values with medians
- Splits into features (X) and target (y)

**Output**: Preprocessed data ready for training. No files saved; data is processed in memory.

**Run**: `python 1_data_preprocessing.py`

### 2. Train Models (`2_train_models.py`)
**Purpose**: Train both ANN (regression) and Logistic Regression (classification) models.

**What it does**:
- Uses preprocessed data
- Trains ANN for predicting continuous diabetes progression score
- Trains Logistic Regression for binary classification (high/low risk)
- Evaluates both models on train/test sets

**Output**:
- Prints regression metrics (RMSE, MAE, R2) for ANN
- Prints classification metrics (accuracy, precision, recall, F1) for Logistic Regression

**Run**: `python 2_train_models.py`

### 3. ANN Results (`3_ann_results.py`)
**Purpose**: Generate detailed results and visualizations for the ANN regression model.

**What it does**:
- Retrains ANN regressor
- Computes performance metrics
- Creates plots: actual vs predicted scatter plot and training loss curve

**Output**:
- `ann_results.txt`: Metrics summary
- `ann_actual_vs_predicted.png`: Scatter plot of predictions
- `ann_loss_curve.png`: Training loss over iterations

**Run**: `python 3_ann_results.py`

### 4. Logistic Regression Results (`4_logistic_results.py`)
**Purpose**: Generate detailed results and visualizations for the Logistic Regression classification model.

**What it does**:
- Retrains Logistic Regression classifier
- Computes performance metrics
- Creates plots: confusion matrix and ROC curve

**Output**:
- `logistic_results.txt`: Metrics summary
- `logistic_confusion_matrix.png`: Confusion matrix heatmap
- `logistic_roc_curve.png`: ROC curve with AUC

**Run**: `python 4_logistic_results.py`

### 5. SHAP Explanation (`5_shap_explanation.py`)
**Purpose**: Apply SHAP (SHapley Additive exPlanations) to explain ANN predictions.

**What it does**:
- Uses SHAP KernelExplainer on the ANN
- Computes feature importance for test samples
- Generates summary and individual prediction plots

**Output**:
- `shap_summary_plot.png`: Feature importance summary
- `shap_force_plot.html`: Interactive explanation for one prediction

**Run**: `python 5_shap_explanation.py`

### 6. LIME Explanation (`6_lime_explanation.py`)
**Purpose**: Apply LIME (Local Interpretable Model-agnostic Explanations) to explain ANN predictions.

**What it does**:
- Uses LIME TabularExplainer on the ANN
- Explains individual predictions by approximating locally
- Generates feature contribution lists

**Output**:
- `lime_explanation.html`: Interactive explanation
- `lime_explanation.txt`: Text summary of feature contributions

**Run**: `python 6_lime_explanation.py`

### 7. Captum Explanation (`7_captum_explanation.py`)
**Purpose**: Apply Captum (PyTorch attribution library) to explain ANN predictions.

**What it does**:
- Trains a PyTorch ANN equivalent
- Computes attributions using Integrated Gradients, Layer Conductance, and Neuron Conductance

**Output**:
- `captum_attributions.txt`: Attribution values for features and layers

**Run**: `python 7_captum_explanation.py`

### 8. Medical Narration (`8_medical_narration.py`)
**Purpose**: Generate a simple, medical-friendly narration from SHAP and LIME explanations.

**What it does**:
- Analyzes SHAP feature importance and LIME explanations
- Maps technical features to medical terms
- Creates a narrative summary for non-technical audiences

**Output**:
- `llm_medical_narration.txt`: Readable medical interpretation

**Run**: `python 8_medical_narration.py`

## Key Insights
- **ANN Performance**: High R2 (~0.999) on test set, indicating good fit for regression
- **Logistic Regression**: ~89% accuracy for binary classification
- **Top Features**: BMI, cholesterol levels, age, blood pressure most influential
- **Medical Takeaway**: Focus on weight management, lipid profiles, and BP control for diabetes prevention

## Running the Full Pipeline
Execute scripts in order:
```bash
python 1_data_preprocessing.py
python 2_train_models.py
python 3_ann_results.py
python 4_logistic_results.py
python 5_shap_explanation.py
python 6_lime_explanation.py
python 7_captum_explanation.py
python 8_medical_narration.py
```

## Files Overview
- **Scripts**: 1-8 numbered Python files
- **Data**: `Ddiabetes_data.csv`
- **Outputs**: `.txt` files (metrics), `.png` files (plots), `.html` files (interactive explanations)

## Conclusion
This project showcases end-to-end ML with emphasis on explainability. The ANN provides accurate predictions, while XAI methods ensure transparency. 
The goal is to predict diabetes progression (regression) or risk classification (binary), while providing clear, medical-friendly explanations of model decisions.
The medical narration makes complex AI insights accessible to healthcare professionals and patients.

