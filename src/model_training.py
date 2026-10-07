"""
Model Training Alias Module
Imports and executes the training pipeline from src/train_model.py.
"""

from src.train_model import run_training_pipeline

def train_and_evaluate_model(
    data_path="data/inventory_demand.csv",
    model_dir="models",
    output_dir="outputs"
):
    return run_training_pipeline(data_path=data_path, model_dir=model_dir, output_dir=output_dir)

if __name__ == "__main__":
    train_and_evaluate_model()
