def log_message(message):
    print(f"[LOG] {message}")

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_random_seed(seed):
    import numpy as np
    import random
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)