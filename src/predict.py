"""
Prediction Module
Loads trained model (.pkl or .joblib) and performs inference on feature inputs.
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd

def load_trained_model(model_path="models/linear_regression_model.pkl"):
    if not os.path.exists(model_path):
        alt_path = "models/linear_regression_model.joblib"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Trained model not found at {model_path}")
            
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = joblib.load(model_path)
        
    return model

def predict_demand(model, feature_df):
    """
    Generates predicted demand array for given feature dataframe.
    Ensures non-negative demand values.
    """
    predictions = model.predict(feature_df)
    predictions = np.maximum(0, predictions)
    return np.round(predictions, 2)

if __name__ == "__main__":
    model = load_trained_model()
    # Sample prediction test
    sample_features = pd.DataFrame([{
        'Year': 2025, 'Month': 6, 'Day': 15, 'Day_of_Week': 2,
        'Lag_1': 110.0, 'Lag_7': 115.0, 'Rolling_Mean_7': 112.5, 'Rolling_Mean_30': 108.0,
        'Current_Stock': 250, 'Lead_Time_Days': 5, 'Previous_Sales': 110.0
    }])
    pred = predict_demand(model, sample_features)
    print(f"Sample Prediction Test Output: {pred[0]} units")
