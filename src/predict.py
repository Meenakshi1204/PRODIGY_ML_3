import joblib
from feature_extractor import extract_features

model = joblib.load("svm_cat_dog.pkl")

def predict_image(image_path):
    features = extract_features(image_path).reshape(1, -1)
    pred = model.predict(features)[0]
    return "Cat" if pred == 0 else "Dog"

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    result = predict_image(image_path)
    print("Prediction:", result)
