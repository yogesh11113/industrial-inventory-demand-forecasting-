"""
Prediction Module
Loads trained model and generates demand predictions for new or selected product data points.
"""

import os
import joblib
import pandas as pd
import numpy as np

def load_trained_model(model_path="models/linear_regression_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

def predict_demand(input_df, model_path="models/linear_regression_model.joblib"):
    model = load_trained_model(model_path)
    feature_cols = [
        'Year', 'Month', 'Day', 'Day_of_Week',
        'Lag_1', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_30',
        'Current_Stock', 'Lead_Time_Days', 'Previous_Sales'
    ]
    
    # Verify features presence
    for col in feature_cols:
        if col not in input_df.columns:
            raise ValueError(f"Missing required feature column: {col}")
            
    X_input = input_df[feature_cols]
    predictions = model.predict(X_input)
    predictions = np.maximum(0, predictions) # Clamp negative predictions
    return np.round(predictions, 2)

if __name__ == "__main__":
    sample_df = pd.read_csv("data/processed/featured_inventory_data.csv").tail(5)
    preds = predict_demand(sample_df)
    sample_df['Predicted_Demand'] = preds
    print("--- SAMPLE PREDICTIONS ---")
    print(sample_df[['Date', 'Product_ID', 'Product_Name', 'Current_Stock', 'Demand', 'Predicted_Demand']])
