from PIL import Image
import numpy as np

IMG_SIZE = (90, 90)

def extract_features(image_input):
    # If input is a path → open normally
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        # If input is a file object (Streamlit upload)
        img = image_input

    img = img.convert("L")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    features = img.flatten()
    return features
