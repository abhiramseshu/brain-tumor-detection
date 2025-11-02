import os
import sys
from src.data.data_loader import load_data
from src.models.cnn_model import CNNModel
from src.models.unet_model import UNetModel
from src.training.train import train_model
from src.training.validate import validate_model

def main():
    # Load the datasets
    train_data, val_data = load_data()

    # Initialize models
    cnn_model = CNNModel()
    unet_model = UNetModel()

    # Train the models
    train_model(cnn_model, train_data)
    train_model(unet_model, train_data)

    # Validate the models
    validate_model(cnn_model, val_data)
    validate_model(unet_model, val_data)

if __name__ == "__main__":
    main()