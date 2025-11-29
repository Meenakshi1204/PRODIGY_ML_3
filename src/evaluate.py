import joblib
from skimage import io, transform
from tkinter import Tk, filedialog

# ---------------------------
# Load the trained SVM model
# ---------------------------
model = joblib.load("svm_model.pkl")

# ---------------------------
# Function to preprocess image
# ---------------------------
def preprocess(image):
    image_resized = transform.resize(image, (64, 64))  # resize to match training size
    return image_resized.flatten()  # flatten to 1D vector

# ---------------------------
# Open File Dialog
# ---------------------------
Tk().withdraw()  # hide Tkinter window

print("📁 Please select an image (JPEG / PNG)")

file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
)

if not file_path:
    print("❌ No file selected")
    exit()

# ---------------------------
# Load and predict
# ---------------------------
print(f"📌 Selected image: {file_path}")

image = io.imread(file_path)
features = preprocess(image)

prediction = model.predict([features])[0]

# ---------------------------
# Output label
# ---------------------------
if prediction == 0:
    print("🐱 The image is predicted as: CAT")
else:
    print("🐶 The image is predicted as: DOG")
