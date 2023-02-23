from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

app = Flask(__name__)

model = tf.keras.models.load_model('./basic-ocr-model')

@app.route('/ping')
def ping():
    return "AI API running"

@app.route("/predict_image", methods=['POST'])
def predict_image():
    file = request.files['image']
    # load image and convert it to gray scale and resize it to 28*28 pixel image
    img = Image.open(file.stream).convert('L').resize((28, 28))
    # convert to 2D array and divide by 255.0 to get float values between 0 and 1
    img_array = np.array(img) / 255.0
    predictions = model.predict( np.array([img_array]) )
    result = np.where(predictions[0] == np.amax(predictions[0]))[0][0]    
    return jsonify({'found number': int(result)})
