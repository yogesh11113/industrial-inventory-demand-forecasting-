"""
Model Training Script
Runnable directly via: python src/train_model.py
Trains a Linear Regression model on industrial inventory demand data using a chronological 80/20 train/test split.
Saves model artifacts, JSON/CSV metrics, and actual vs. predicted visualization plots.
"""

import os
import sys
import json
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ensure parent directory is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features

def run_training_pipeline(
    data_path="data/inventory_demand.csv",
    model_dir="models",
    output_dir="outputs"
):
    print("=" * 50)
    print("INDUSTRIAL INVENTORY DEMAND FORECASTING - TRAINING")
    print("=" * 50)
    
    # 1. Load Dataset
    print("\nLoading dataset...")
    df_raw = load_data(data_path)
    total_records = len(df_raw)
    print(f"Dataset loaded successfully.")
    print(f"File path: {data_path}")
    print(f"Total samples: {total_records}")
    
    # 2. Data Preprocessing
    print("\nPreprocessing data...")
    df_clean, summary = preprocess_data(df_raw)
    print(f"Preprocessing completed. (Duplicates removed: {summary['duplicates_removed']})")
    
    # 3. Feature Engineering
    print("\nCreating features...")
    df_featured, feature_cols, target_col = create_features(df_clean)
    print(f"Features created successfully ({len(feature_cols)} features).")
    print(f"Target column: {target_col}")
    
    # 4. Chronological Train/Test Split (80% Train, 20% Test)
    print("\nSplitting dataset (80% Train / 20% Test)...")
    split_idx = int(len(df_featured) * 0.8)
    
    X = df_featured[feature_cols]
    y = df_featured[target_col]
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_df = df_featured.iloc[split_idx:].copy()
    
    train_count = len(X_train)
    test_count = len(X_test)
    print(f"Training samples: {train_count}")
    print(f"Testing samples: {test_count}")
    
    # 5. Model Training (Genuine model.fit execution)
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Executed: model.fit(X_train, y_train)")
    print("Model training completed.")
    
    # 6. Evaluation & Predictions
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred) # Demand cannot be negative
    test_df['Predicted_Demand'] = np.round(y_pred, 2)
    
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    metrics = {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4),
        "Algorithm": "Linear Regression",
        "Train_Samples": train_count,
        "Test_Samples": test_count,
        "Total_Samples": total_records,
        "Features": feature_cols,
        "Target": target_col,
        "Trained_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"\nMAE: {metrics['MAE']}")
    print(f"MSE: {metrics['MSE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"R² Score: {metrics['R2_Score']}")
    
    # 7. Save Artifacts
    print("\nSaving artifacts...")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    
    pkl_model_path = os.path.join(model_dir, "linear_regression_model.pkl")
    joblib_model_path = os.path.join(model_dir, "linear_regression_model.joblib")
    
    with open(pkl_model_path, "wb") as f:
        pickle.dump(model, f)
    joblib.dump(model, joblib_model_path)
    
    print(f"Model saved successfully: {pkl_model_path}")
    print(f"Model saved successfully: {joblib_model_path}")
    
    # Save Metrics (JSON & CSV)
    json_metrics_path = os.path.join(output_dir, "evaluation", "model_metrics.json")
    csv_metrics_path = os.path.join(output_dir, "evaluation", "evaluation_metrics.csv")
    
    with open(json_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    pd.DataFrame([metrics]).to_csv(csv_metrics_path, index=False)
    
    print(f"Metrics saved successfully: {json_metrics_path}")
    print(f"Metrics saved successfully: {csv_metrics_path}")
    
    # Save Predictions CSV
    test_df.to_csv(os.path.join(output_dir, "predictions", "test_predictions.csv"), index=False)
    
    # 8. Generate Visualizations & EDA Plots
    save_visualizations(test_df, y_test, y_pred, df_featured, output_dir)
    
    print("=" * 50)
    print("TRAINING PIPELINE SUCCESSFULLY EXECUTED")
    print("=" * 50)
    
    return model, metrics, test_df

def save_visualizations(test_df, y_test, y_pred, full_df, output_dir):
    graphs_dir = os.path.join(output_dir, "graphs")
    
    # Plot 1: Actual vs Predicted Demand
    plt.figure(figsize=(12, 5))
    plt.plot(test_df['Date'].iloc[:150], y_test.iloc[:150], label='Actual Demand', color='#2563EB', linewidth=2)
    plt.plot(test_df['Date'].iloc[:150], y_pred[:150], label='Predicted Demand (Linear Regression)', color='#D97706', linestyle='--', linewidth=2)
    plt.title("Actual vs Predicted Industrial Demand (Test Subset)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Demand Units", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "actual_vs_predicted.png"), dpi=300)
    plt.savefig(os.path.join(graphs_dir, "8_actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # Plot 2: Demand Trend Over Time
    plt.figure(figsize=(12, 5))
    sample_prod = full_df[full_df['Product_ID'] == 'PRD_001']
    plt.plot(sample_prod['Date'], sample_prod['Demand'], color='#059669', linewidth=2, label='Daily Demand (Hydraulic Valve A1)')
    plt.title("Demand Trend Over Time (Product PRD_001)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Demand Units", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "demand_trend.png"), dpi=300)
    plt.close()
    
    # Plot 3: Demand Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(full_df['Demand'], kde=True, color='#2563EB', bins=30)
    plt.title("Demand Distribution Across Industrial SKUs", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Demand Units", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "demand_distribution.png"), dpi=300)
    plt.close()
    
    # Plot 4: Current Stock vs Demand
    plt.figure(figsize=(10, 5))
    plt.scatter(full_df['Current_Stock'].iloc[:300], full_df['Demand'].iloc[:300], alpha=0.6, color='#DC2626')
    plt.title("Current Stock Level vs Daily Demand", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Current Stock Level (Units)", fontsize=11)
    plt.ylabel("Daily Demand (Units)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "current_stock_vs_demand.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_training_pipeline()
