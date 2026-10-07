# 🏭 Industrial Inventory Demand Forecasting & Stock Reorder System
**Problem-Based Learning (PBL) Project Review-II (30% Progress Milestone)**

---

## 📌 Project Overview
This project delivers an end-to-end Machine Learning pipeline and interactive Streamlit application for **Industrial Inventory Demand Forecasting**. By analyzing historical demand, consumption rates, stock depletion, and supplier lead times across 10 industrial SKUs, the system predicts daily product demand using supervised machine learning and automatically calculates recommended safety stock reorder quantities.

---

## 📂 Project Directory Structure

```text
industrial-inventory-demand-forecasting/
├── data/
│   ├── inventory_demand.csv               # Primary dataset (openable in VS Code / Excel)
│   ├── processed_inventory_demand.csv   # Preprocessed inventory records
│   └── processed/
│       └── featured_inventory_data.csv    # Engineered temporal & lag features
├── notebooks/
│   └── inventory_demand_forecasting.ipynb # Executed end-to-end ML notebook for faculty
├── src/
│   ├── data_preprocessing.py              # Data cleaning, missing values & sorting
│   ├── feature_engineering.py             # Lag & rolling window feature creation
│   ├── train_model.py                      # Training script with model.fit() & evaluation
│   ├── predict.py                         # Model loading & inference module
│   └── reorder_recommendation.py          # Safety stock reorder formulation
├── models/
│   ├── linear_regression_model.pkl        # Trained pickle model artifact
│   └── linear_regression_model.joblib     # Trained joblib model artifact
├── outputs/
│   ├── graphs/
│   │   ├── actual_vs_predicted.png        # Actual vs Predicted demand plot
│   │   ├── demand_trend.png               # Time-series demand trend
│   │   └── demand_distribution.png        # Demand histogram distribution
│   ├── evaluation/
│   │   ├── model_metrics.json             # MAE, MSE, RMSE, R² metrics
│   │   └── evaluation_metrics.csv         # Tabular evaluation metrics
│   └── predictions/
│       └── test_predictions.csv           # Test set prediction log
├── app/
│   ├── app.py                             # Streamlit dashboard application
│   └── streamlit_app.py                   # Streamlit app launcher
├── requirements.txt                       # Project dependencies
└── README.md                              # Technical documentation
```

---

## 🔬 Machine Learning Workflow

```text
DATASET (data/inventory_demand.csv)
   ↓
PREPROCESSING (src/data_preprocessing.py)
   ↓
FEATURE ENGINEERING (src/feature_engineering.py)
   ↓
CHRONOLOGICAL SPLIT (80% Train / 20% Test)
   ↓
MODEL TRAINING (model.fit(X_train, y_train))
   ↓
MODEL SAVING (models/linear_regression_model.pkl)
   ↓
MODEL EVALUATION (MAE, MSE, RMSE, R² Score)
   ↓
REORDER RECOMMENDATION LOGIC
   ↓
STREAMLIT DASHBOARD (python -m streamlit run app/app.py)
```

---

## ⚡ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Training Script
Execute the complete ML pipeline to train the model, calculate evaluation metrics, and generate plots:
```bash
python src/train_model.py
```

Expected Terminal Output:
```text
==================================================
INDUSTRIAL INVENTORY DEMAND FORECASTING - TRAINING
==================================================

Loading dataset...
Dataset loaded successfully.
File path: data/inventory_demand.csv
Total samples: 3650

Preprocessing data...
Preprocessing completed. (Duplicates removed: 0)

Creating features...
Features created successfully (11 features).
Target column: Demand

Splitting dataset (80% Train / 20% Test)...
Training samples: 2920
Testing samples: 730

Training Linear Regression model...
Executed: model.fit(X_train, y_train)
Model training completed.

Evaluating model...

MAE: 21.1826
MSE: 881.566
RMSE: 29.6912
R² Score: 0.9037

Saving artifacts...
Model saved successfully: models/linear_regression_model.pkl
Model saved successfully: models/linear_regression_model.joblib
Metrics saved successfully: outputs/evaluation/model_metrics.json
==================================================
```

### 3. Launch the Interactive Streamlit Application
```bash
python -m streamlit run app/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🎯 Current Milestone (30% PBL Review-II)
- [x] Primary Dataset Generation (`data/inventory_demand.csv`)
- [x] Data Preprocessing Module (`src/data_preprocessing.py`)
- [x] Feature Engineering without Data Leakage (`src/feature_engineering.py`)
- [x] Chronological 80/20 Train/Test Split
- [x] Supervised Linear Regression Training (`model.fit()`)
- [x] Artifact Saving (`models/linear_regression_model.pkl`)
- [x] Evaluation Metrics Generation (`outputs/evaluation/model_metrics.json`)
- [x] Executed Jupyter Notebook (`notebooks/inventory_demand_forecasting.ipynb`)
- [x] Interactive Streamlit Dashboard with "Live Retrain Model" Button (`app/app.py`)

---

## ⏳ Upcoming Milestone Scope (Review-III & Final)
- [ ] **Phase 2:** FB Prophet Time-Series Modeling for Seasonality & Trend Changepoint Detection.
- [ ] **Phase 3:** Deep Learning LSTM Neural Networks for Multi-Step Sequence-to-Sequence Forecasting.
- [ ] **Phase 4:** Model Comparison Framework & Advanced Multi-Echelon Reorder Logic.
