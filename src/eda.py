"""
Exploratory Data Analysis (EDA) Module
Generates high-quality, professional visualizations for inventory and demand patterns and saves them in outputs/graphs/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_plots(data_path="data/processed/preprocessed_inventory_data.csv", output_dir="outputs/graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Styling configuration for presentation suitability
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_palette("muted")
    
    # 1. Demand Over Time
    plt.figure(figsize=(12, 5))
    daily_demand = df.groupby('Date')['Demand'].sum().reset_index()
    plt.plot(daily_demand['Date'], daily_demand['Demand'], color='#1f77b4', linewidth=1.8, label='Total Daily Demand')
    plt.title("Industrial Total Inventory Demand Over Time (2025)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Demand Units", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_demand_over_time.png"), dpi=300)
    plt.close()
    
    # 2. Sales Quantity Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Sales_Quantity'], kde=True, color='#2ca02c', bins=30)
    plt.title("Distribution of Daily Sales Quantity across Products", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Sales Quantity", fontsize=12)
    plt.ylabel("Frequency / Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_sales_quantity_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Demand by Product Category
    plt.figure(figsize=(10, 5))
    cat_demand = df.groupby('Category')['Demand'].sum().sort_values(ascending=False).reset_index()
    sns.barplot(data=cat_demand, x='Category', y='Demand', palette='Blues_r')
    plt.title("Total Demand Breakdown by Product Category", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Total Demand", fontsize=12)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_demand_by_product_category.png"), dpi=300)
    plt.close()
    
    # 4. Current Stock vs Demand
    plt.figure(figsize=(10, 5))
    sample_df = df.sample(n=min(500, len(df)), random_state=42)
    plt.scatter(sample_df['Current_Stock'], sample_df['Demand'], alpha=0.6, color='#ff7f0e', edgecolors='w', s=50)
    plt.title("Current Stock vs. Daily Product Demand", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Current Stock Level", fontsize=12)
    plt.ylabel("Product Demand", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_current_stock_vs_demand.png"), dpi=300)
    plt.close()
    
    # 5. Monthly Demand Trend
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    df['Month_Num'] = df['Date'].dt.month
    monthly_trend = df.groupby(['Month_Num', 'Month_Name'])['Demand'].sum().reset_index()
    monthly_trend = monthly_trend.sort_values('Month_Num')
    
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_trend['Month_Name'], monthly_trend['Demand'], marker='o', linewidth=2.5, color='#d62728')
    plt.title("Monthly Demand Trend Analysis (Seasonal Variation)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Total Monthly Demand", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_monthly_demand_trend.png"), dpi=300)
    plt.close()
    
    # 6. Top 10 Products by Demand
    plt.figure(figsize=(11, 5))
    top_prod = df.groupby('Product_Name')['Demand'].sum().sort_values(ascending=True).reset_index()
    plt.barh(top_prod['Product_Name'], top_prod['Demand'], color='#9467bd')
    plt.title("Top Products Ranked by Total Cumulative Demand", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Cumulative Demand", fontsize=12)
    plt.ylabel("Product Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_top_products_by_demand.png"), dpi=300)
    plt.close()
    
    # 7. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    numeric_cols = ['Sales_Quantity', 'Consumption', 'Current_Stock', 'Lead_Time_Days', 'Unit_Price', 'Demand']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", linewidths=0.8, cbar=True)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "7_correlation_heatmap.png"), dpi=300)
    plt.close()

    print(f"All 7 EDA visualizations generated and saved in '{output_dir}/'")

if __name__ == "__main__":
    generate_eda_plots()
