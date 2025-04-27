from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import requests

app = Flask(__name__)
CORS(app)

# Define model download URL (shareable link from Google Drive)
GOOGLE_DRIVE_FILE_ID = '1d_zmXyypxBe7h5rh07IXgjxgpwMRzeyu'  # replace with your file ID
MODEL_PATH = 'model/model.h5'

# Ensure that the directory to store the model exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def download_model():
    """Download the model from Google Drive if not already downloaded."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            print("Error: Model download failed!")
    else:
        print("Model already exists locally.")

# Download model first (only if not already present)
download_model()

# Load the trained model once when the server starts
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Define class names
class_names = [
    "actinic keratosis", "basal cell carcinoma", "dermatofibroma",
    "melanoma", "nevus", "pigmented benign keratosis",
    "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion"
]

# Define expected input shape
input_shape = (180, 180)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    try:
        # Load and preprocess the image
        image = Image.open(file).convert('RGB')
        image = image.resize(input_shape)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, axis=0)

        print(f"Image shape after processing: {image.shape}")

        # Predict using the model
        prediction = model.predict(image)
        print(f"Raw model prediction: {prediction}")

        # Apply softmax
        prediction = tf.nn.softmax(prediction[0]).numpy()
        print(f"Softmax probabilities: {prediction}")

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[predicted_class_index])

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
