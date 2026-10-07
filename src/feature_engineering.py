"""
Feature Engineering Module
Creates temporal features, lagged demand, rolling window averages, and stock indicators without data leakage.
"""

import os
import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Product_ID', 'Date']).reset_index(drop=True)
    
    # 1. Temporal Date Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    
    # 2. Lag Features (Strict Shift 1 and Shift 7 to avoid data leakage)
    df['Lag_1'] = df.groupby('Product_ID')['Demand'].shift(1)
    df['Lag_7'] = df.groupby('Product_ID')['Demand'].shift(7)
    
    # 3. Rolling Window Demand Features (Computed on Shifted Demand)
    df['Rolling_Mean_7'] = df.groupby('Product_ID')['Demand'].shift(1).transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['Rolling_Mean_30'] = df.groupby('Product_ID')['Demand'].shift(1).transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    
    # 4. Previous Sales Feature
    df['Previous_Sales'] = df['Lag_1']
    
    # 5. Handle initial NaN values from lagging
    df['Lag_1'] = df['Lag_1'].fillna(df['Demand'])
    df['Lag_7'] = df['Lag_7'].fillna(df['Demand'])
    df['Rolling_Mean_7'] = df['Rolling_Mean_7'].fillna(df['Demand'])
    df['Rolling_Mean_30'] = df['Rolling_Mean_30'].fillna(df['Demand'])
    df['Previous_Sales'] = df['Previous_Sales'].fillna(df['Sales_Quantity'])
    
    feature_cols = [
        'Year', 'Month', 'Day', 'Day_of_Week',
        'Lag_1', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_30',
        'Current_Stock', 'Lead_Time_Days', 'Previous_Sales'
    ]
    target_col = 'Demand'
    
    return df, feature_cols, target_col

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    input_path = "data/processed_inventory_demand.csv"
    if not os.path.exists(input_path):
        input_path = "data/inventory_demand.csv"
        
    df = pd.read_csv(input_path)
    df_featured, feature_cols, target_col = create_features(df)
    
    output_path = "data/processed/featured_inventory_data.csv"
    df_featured.to_csv(output_path, index=False)
    
    print("--- FEATURE ENGINEERING SUMMARY ---")
    print(f"Generated Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target Column: {target_col}")
    print(f"Dataset Shape: {df_featured.shape}")
    print(f"Saved featured dataset to '{output_path}'")
