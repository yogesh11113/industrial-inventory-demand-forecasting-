import os
import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Industrial Inventory Demand Forecasting\n",
    "## PBL Project Review-II (30% Progress Milestone Notebook)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "from src.generate_data import generate_synthetic_data\n",
    "from src.data_preprocessing import preprocess_data\n",
    "from src.eda import generate_eda_plots\n",
    "from src.feature_engineering import create_features\n",
    "from src.model_training import train_and_evaluate_model\n",
    "from src.reorder_recommendation import generate_reorder_recommendation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Dataset Generation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_raw = generate_synthetic_data()\n",
    "df_raw.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_processed, summary = preprocess_data(df_raw)\n",
    "print('Preprocessing Summary:', summary)\n",
    "df_processed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_featured = create_features(df_processed)\n",
    "df_featured.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Model Training & Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model, metrics, test_df = train_and_evaluate_model()\n",
    "print('Metrics:', metrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(test_df['Date'].iloc[:100], test_df['Demand'].iloc[:100], label='Actual Demand')\n",
    "plt.plot(test_df['Date'].iloc[:100], test_df['Predicted_Demand'].iloc[:100], label='Predicted Demand', linestyle='--')\n",
    "plt.title('Actual vs Predicted Demand')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Reorder Recommendation Prototype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "rec = generate_reorder_recommendation(predicted_demand=120, current_stock=80, lead_time_days=5)\n",
    "print('Recommendation Status:', rec['status'])\n",
    "print('Recommended Reorder Quantity:', rec['recommended_reorder_quantity'])"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/inventory_demand_forecasting.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)
print('Jupyter Notebook successfully created at notebooks/inventory_demand_forecasting.ipynb')
