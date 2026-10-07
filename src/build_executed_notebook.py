"""
Script to create and execute notebooks/inventory_demand_forecasting.ipynb with full cell outputs.
"""

import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

def create_and_run_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell("# 🏭 Industrial Inventory Demand Forecasting\n### CSE PBL Project Review-II (30% Progress Milestone Notebook)\n\nThis notebook demonstrates the end-to-end Machine Learning pipeline from data loading, preprocessing, feature engineering, chronological train/test split, model training using `model.fit()`, evaluation, and artifact saving."),
        
        # Step 1: Import Libraries
        new_markdown_cell("### Step 1: Import Libraries"),
        new_code_cell("""import os
import sys
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(".."))
print("All required libraries imported successfully.")"""),

        # Step 2: Load Dataset
        new_markdown_cell("### Step 2: Load Dataset"),
        new_code_cell("""data_path = "../data/inventory_demand.csv"
if not os.path.exists(data_path):
    data_path = "data/inventory_demand.csv"

df_raw = pd.read_csv(data_path)
print(f"Loaded dataset from '{data_path}' successfully.")"""),

        # Step 3: Display Dataset Preview
        new_markdown_cell("### Step 3: Display Dataset Preview"),
        new_code_cell("""df_raw.head()"""),

        # Step 4: Show Number of Rows and Columns
        new_markdown_cell("### Step 4: Show Dataset Shape (Rows and Columns)"),
        new_code_cell("""print(f"Total Rows (Records): {df_raw.shape[0]}")
print(f"Total Columns (Features): {df_raw.shape[1]}")
print(f"Columns: {list(df_raw.columns)}")"""),

        # Step 5: Show Missing Values
        new_markdown_cell("### Step 5: Check Missing Values"),
        new_code_cell("""missing_values = df_raw.isnull().sum()
print("Missing values per column:")
print(missing_values)"""),

        # Step 6: Data Preprocessing
        new_markdown_cell("### Step 6: Data Preprocessing"),
        new_code_cell("""from src.data_preprocessing import preprocess_data

df_clean, summary = preprocess_data(df_raw)
print("Preprocessing Summary:", summary)
df_clean.head()"""),

        # Step 7: Exploratory Data Analysis (EDA)
        new_markdown_cell("### Step 7: Exploratory Data Analysis (EDA)"),
        new_code_cell("""plt.figure(figsize=(10, 4))
sample_prod = df_clean[df_clean['Product_ID'] == 'PRD_001']
plt.plot(pd.to_datetime(sample_prod['Date']), sample_prod['Demand'], color='#2563EB', linewidth=2)
plt.title("Historical Demand Trend (PRD_001 - Hydraulic Valve A1)")
plt.xlabel("Date")
plt.ylabel("Demand Units")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()"""),

        # Step 8: Feature Engineering
        new_markdown_cell("### Step 8: Feature Engineering"),
        new_code_cell("""from src.feature_engineering import create_features

df_featured, feature_cols, target_col = create_features(df_clean)
print(f"Engineered Features ({len(feature_cols)}):", feature_cols)
print("Target Column:", target_col)
df_featured[feature_cols + [target_col]].head()"""),

        # Step 9: Define X (Features) and y (Target)
        new_markdown_cell("### Step 9: Define X (Features Matrix) and y (Target Vector)"),
        new_code_cell("""X = df_featured[feature_cols]
y = df_featured[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)"""),

        # Step 10: Chronological Train / Test Split
        new_markdown_cell("### Step 10: Chronological Train / Test Split (80% Train, 20% Test)"),
        new_code_cell("""split_idx = int(len(df_featured) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_df = df_featured.iloc[split_idx:].copy()

print(f"Training samples (X_train): {len(X_train)}")
print(f"Testing samples (X_test)  : {len(X_test)}")"""),

        # Step 11: Create Linear Regression Model
        new_markdown_cell("### Step 11: Instantiate Linear Regression Model"),
        new_code_cell("""model = LinearRegression()
print("Linear Regression model instantiated:", model)"""),

        # Step 12: Actual Model Training Line
        new_markdown_cell("### Step 12: Actual Model Training (`model.fit(X_train, y_train)`)"),
        new_code_cell("""# EXECUTING ACTUAL MODEL TRAINING LINE
model.fit(X_train, y_train)
print("Model training completed via model.fit(X_train, y_train)!")"""),

        # Step 13: Generate Predictions
        new_markdown_cell("### Step 13: Generate Predictions on Test Set"),
        new_code_cell("""y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred) # Non-negative demand constraint
test_df['Predicted_Demand'] = np.round(y_pred, 2)
print("Generated predictions for test samples.")
test_df[['Date', 'Product_ID', 'Current_Stock', 'Lead_Time_Days', 'Demand', 'Predicted_Demand']].head()"""),

        # Step 14 - 17: Evaluation Metrics (MAE, MSE, RMSE, R²)
        new_markdown_cell("### Step 14 - 17: Calculate Evaluation Metrics (MAE, MSE, RMSE, R² Score)"),
        new_code_cell("""mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Step 14 - MAE     : {mae:.4f}")
print(f"Step 15 - MSE     : {mse:.4f}")
print(f"Step 16 - RMSE    : {rmse:.4f}")
print(f"Step 17 - R² Score: {r2:.4f} ({r2*100:.2f}% Variance Explained)")"""),

        # Step 18: Show Actual vs Predicted Graph
        new_markdown_cell("### Step 18: Show Actual vs Predicted Demand Graph"),
        new_code_cell("""plt.figure(figsize=(12, 5))
plt.plot(pd.to_datetime(test_df['Date']).iloc[:120], y_test.iloc[:120], label='Actual Demand', color='#2563EB', linewidth=2)
plt.plot(pd.to_datetime(test_df['Date']).iloc[:120], y_pred[:120], label='Predicted Demand', color='#D97706', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Demand (Linear Regression Model)")
plt.xlabel("Date")
plt.ylabel("Demand Units")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()"""),

        # Step 19: Save Trained Model
        new_markdown_cell("### Step 19: Save Trained Model"),
        new_code_cell("""os.makedirs("../models", exist_ok=True)
model_path = "../models/linear_regression_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Saved trained model successfully to '{model_path}'.")"""),

        # Step 20: Demonstrate Loading the Trained Model
        new_markdown_cell("### Step 20: Demonstrate Loading the Saved Model"),
        new_code_cell("""with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

print("Loaded model object successfully:", loaded_model)"""),

        # Step 21: Generate a Sample Prediction
        new_markdown_cell("### Step 21: Generate a Sample Prediction Using Loaded Model"),
        new_code_cell("""sample_input = pd.DataFrame([{
    'Year': 2025, 'Month': 7, 'Day': 20, 'Day_of_Week': 6,
    'Lag_1': 140.0, 'Lag_7': 135.0, 'Rolling_Mean_7': 138.0, 'Rolling_Mean_30': 130.0,
    'Current_Stock': 180, 'Lead_Time_Days': 5, 'Previous_Sales': 140.0
}])

sample_pred = loaded_model.predict(sample_input)[0]
print(f"Sample ML Input Features: Year=2025, Month=7, Lag_1=140.0, Current_Stock=180")
print(f"Predicted Daily Demand  : {max(0, round(float(sample_pred), 2))} units")""")
    ]
    
    nb.cells = cells
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = "notebooks/inventory_demand_forecasting.ipynb"
    
    # Execute notebook using ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': 'notebooks'}})
    except Exception as e:
        print(f"Warning during execution: {e}")
        
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"Executed Jupyter Notebook saved at '{nb_path}' successfully.")

if __name__ == "__main__":
    create_and_run_notebook()
