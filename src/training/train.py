from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
from src.data.data_loader import load_data
from src.data.preprocessing import preprocess_images

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    model = Model(inputs, outputs)
    return model

def train_model(model, train_data, train_labels, val_data, val_labels, epochs=50, batch_size=32):
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(train_data, train_labels, 
              validation_data=(val_data, val_labels), 
              epochs=epochs, 
              batch_size=batch_size, 
              callbacks=[checkpoint, early_stopping])

def main():
    input_shape = (128, 128, 1)  # Example input shape for grayscale images
    train_data, train_labels, val_data, val_labels = load_data()
    train_data = preprocess_images(train_data)
    val_data = preprocess_images(val_data)

    model = create_model(input_shape)
    train_model(model, train_data, train_labels, val_data, val_labels)

if __name__ == "__main__":
    main()