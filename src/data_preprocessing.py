"""
Data Preprocessing Module
Handles loading, missing value checks, duplicate removal, date conversion, chronological sorting, and categorical encoding.
Saves outputs to data/processed_inventory_demand.csv and data/processed/preprocessed_inventory_data.csv.
"""

import os
import pandas as pd
import numpy as np

def load_data(file_path="data/inventory_demand.csv"):
    if not os.path.exists(file_path):
        # Fallback check
        alt_path = "data/raw/industrial_inventory_data.csv"
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Inventory dataset file not found at {file_path}")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    initial_shape = df.shape
    
    # 1. Missing values
    missing_before = df.isnull().sum().to_dict()
    df = df.dropna()
    
    # 2. Duplicate records
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        
    # 3. Date conversion & chronological sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Product_ID', 'Date']).reset_index(drop=True)
    
    # 4. Handle obvious invalid/negative values
    if 'Sales_Quantity' in df.columns:
        df['Sales_Quantity'] = df['Sales_Quantity'].apply(lambda x: max(0, x))
    if 'Consumption' in df.columns:
        df['Consumption'] = df['Consumption'].apply(lambda x: max(0, x))
    if 'Current_Stock' in df.columns:
        df['Current_Stock'] = df['Current_Stock'].apply(lambda x: max(0, x))
    if 'Demand' in df.columns:
        df['Demand'] = df['Demand'].apply(lambda x: max(0, x))
        
    final_shape = df.shape
    
    summary = {
        "initial_shape": initial_shape,
        "final_shape": final_shape,
        "duplicates_removed": int(duplicates_count),
        "missing_values": missing_before
    }
    
    return df, summary

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    raw_df = load_data()
    processed_df, summary = preprocess_data(raw_df)
    
    output_path1 = "data/processed_inventory_demand.csv"
    output_path2 = "data/processed/preprocessed_inventory_data.csv"
    
    processed_df.to_csv(output_path1, index=False)
    processed_df.to_csv(output_path2, index=False)
    
    print("--- PREPROCESSING SUMMARY ---")
    print(f"Initial Shape     : {summary['initial_shape']}")
    print(f"Final Shape       : {summary['final_shape']}")
    print(f"Duplicates Removed: {summary['duplicates_removed']}")
    print(f"Saved processed dataset to '{output_path1}' and '{output_path2}'")
