"""
Reorder Recommendation System Module
Determines inventory reorder status and calculates recommended reorder quantity based on predicted demand, current stock, and lead time.
"""

import pandas as pd
import numpy as np

def generate_reorder_recommendation(predicted_demand, current_stock, lead_time_days, safety_factor=1.2):
    """
    Basic Prototype Reorder Logic (30% Milestone):
    - Required Planning Period Stock = Predicted Daily Demand * (Lead Time Days * Safety Factor)
    - If Current Stock < Required Planning Period Stock:
          Status = "REORDER REQUIRED"
          Recommended Quantity = Required Stock - Current Stock
      Else:
          Status = "STOCK SUFFICIENT"
          Recommended Quantity = 0
    """
    required_stock = float(predicted_demand) * max(1, lead_time_days) * safety_factor
    
    if current_stock < required_stock:
        status = "REORDER REQUIRED"
        reorder_qty = int(np.ceil(required_stock - current_stock))
    else:
        status = "STOCK SUFFICIENT"
        reorder_qty = 0
        
    return {
        "status": status,
        "recommended_reorder_quantity": reorder_qty,
        "required_safety_stock_threshold": int(np.ceil(required_stock))
    }

def evaluate_inventory_dataset(df):
    results = []
    for idx, row in df.iterrows():
        rec = generate_reorder_recommendation(
            predicted_demand=row['Predicted_Demand'],
            current_stock=row['Current_Stock'],
            lead_time_days=row['Lead_Time_Days']
        )
        results.append({
            "Product_ID": row['Product_ID'],
            "Product_Name": row['Product_Name'],
            "Current_Stock": row['Current_Stock'],
            "Predicted_Demand": row['Predicted_Demand'],
            "Lead_Time_Days": row['Lead_Time_Days'],
            "Recommendation_Status": rec['status'],
            "Recommended_Reorder_Qty": rec['recommended_reorder_quantity']
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    sample_rec = generate_reorder_recommendation(predicted_demand=120, current_stock=80, lead_time_days=5)
    print("--- SAMPLE REORDER RECOMMENDATION TEST ---")
    print(f"Predicted Demand: 120 units | Current Stock: 80 units | Lead Time: 5 days")
    print(f"Recommendation Status : {sample_rec['status']}")
    print(f"Recommended Reorder Qty: {sample_rec['recommended_reorder_quantity']} units")
