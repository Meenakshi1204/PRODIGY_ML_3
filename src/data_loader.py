# src/data_loader.py
import os
from feature_extractor import extract_features
import numpy as np

def load_dataset(cat_dir, dog_dir):
    X = []
    y = []

    # Load cat images
    for file in os.listdir(cat_dir):
        path = os.path.join(cat_dir, file)
        X.append(extract_features(path))
        y.append(0)   # Cat = 0

    # Load dog images
    for file in os.listdir(dog_dir):
        path = os.path.join(dog_dir, file)
        X.append(extract_features(path))
        y.append(1)   # Dog = 1

    return np.array(X), np.array(y)
