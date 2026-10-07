"""
Synthetic Industrial Inventory & Demand Data Generator
Generates realistic historical inventory, sales, consumption, and demand data across multiple products and dates.
Saves primary CSV dataset to data/inventory_demand.csv.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_days=365, num_products=10, seed=42):
    np.random.seed(seed)
    
    products = [
        {"id": f"PRD_{i+1:03d}", "name": name, "category": cat, "base_demand": bd, "unit_price": up, "lead_time": lt}
        for i, (name, cat, bd, up, lt) in enumerate([
            ("Hydraulic Valve A1", "Hydraulics", 120, 250.0, 5),
            ("Pneumatic Cylinder C2", "Pneumatics", 85, 180.0, 4),
            ("Industrial Bearing B7", "Bearings", 200, 45.0, 3),
            ("Electric Motor M5", "Electrical", 40, 850.0, 10),
            ("Steel Fastener Set", "Hardware", 500, 12.0, 2),
            ("Control PCB Board", "Electronics", 65, 320.0, 7),
            ("Thermal Sensor T8", "Electronics", 150, 95.0, 4),
            ("Conveyor Belt B12", "Mechanical", 30, 1200.0, 14),
            ("Filter Cartridge F3", "Maintenance", 250, 28.0, 3),
            ("Lubricant Drum L1", "Maintenance", 90, 210.0, 6),
        ])
    ]
    
    start_date = pd.to_datetime("2025-01-01")
    date_range = pd.date_range(start=start_date, periods=num_days, freq="D")
    
    suppliers = ["Apex Industrial Solutions", "Vortex Engineering Supplies", "Global Components Co", "OmniTech Distribution"]
    
    records = []
    
    for prod in products:
        current_stock = np.random.randint(150, 400)
        
        for d_idx, current_date in enumerate(date_range):
            day_of_year = current_date.dayofyear
            day_of_week = current_date.dayofweek
            
            seasonality = 1.0 + 0.25 * np.sin(2 * np.pi * day_of_year / 365) + 0.15 * np.cos(4 * np.pi * day_of_year / 365)
            weekday_factor = 0.7 if day_of_week in [5, 6] else 1.1
            noise = np.random.normal(1.0, 0.12)
            
            raw_demand = prod["base_demand"] * seasonality * weekday_factor * noise
            demand = max(5, int(round(raw_demand)))
            
            sales_quantity = min(demand, current_stock)
            consumption = int(round(sales_quantity * np.random.uniform(0.95, 1.05)))
            
            if current_stock < demand * 0.5:
                stock_level = "Critical"
            elif current_stock < demand * 1.2:
                stock_level = "Low"
            elif current_stock > demand * 3.0:
                stock_level = "Overstocked"
            else:
                stock_level = "Normal"
                
            supplier = np.random.choice(suppliers)
            
            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Product_ID": prod["id"],
                "Product_Name": prod["name"],
                "Category": prod["category"],
                "Sales_Quantity": sales_quantity,
                "Consumption": consumption,
                "Current_Stock": current_stock,
                "Lead_Time_Days": prod["lead_time"],
                "Supplier": supplier,
                "Unit_Price": prod["unit_price"],
                "Stock_Level": stock_level,
                "Demand": demand
            })
            
            current_stock -= sales_quantity
            if current_stock < prod["base_demand"] * 1.5:
                current_stock += int(prod["base_demand"] * np.random.uniform(3.5, 5.0))
                
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    df = generate_synthetic_data(num_days=365, num_products=10)
    
    primary_path = "data/inventory_demand.csv"
    raw_path = "data/raw/industrial_inventory_data.csv"
    
    df.to_csv(primary_path, index=False)
    df.to_csv(raw_path, index=False)
    
    print(f"Dataset successfully saved to '{primary_path}' and '{raw_path}' with {len(df)} records.")
