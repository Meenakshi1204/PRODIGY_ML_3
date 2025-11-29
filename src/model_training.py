# src/model_training.py
from sklearn.svm import SVC
from data_loader import load_dataset
import joblib

cat_path = "data/train/cat"
dog_path = "data/train/dog"

print("Loading dataset...")
X, y = load_dataset(cat_path, dog_path)

print("Training model...")
model = SVC(kernel="linear", probability=True)
model.fit(X, y)

joblib.dump(model, "svm_cat_dog.pkl")
print("Model saved as svm_cat_dog.pkl")
