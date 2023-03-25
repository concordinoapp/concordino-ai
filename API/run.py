from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from tensorflow.keras import layers
from app.utils import create_temp_upload_dir
from app.encoding import encode_single_sample, decode_cnn_ocr_prediction_model
from app.model import load_model, load_prediciton_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
UPLOADED_FILE_DIRECTORY = os.path.join(CURRENT_SCRIPT_PATH, "__uploaded_files__/")
MODEL_PATH = os.path.join(CURRENT_SCRIPT_PATH, "app/assets/models/CNN-MODEL-V4")


app = Flask(__name__)
create_temp_upload_dir(UPLOADED_FILE_DIRECTORY)
prediction_model = load_prediciton_model(load_model(MODEL_PATH)) # Prediciton model needs real model to load


@app.route('/ping')
def ping():
    return "AI API running"

@app.route('/cnn-ocr-model/predict_image', methods=['POST'])
def ocr_captcha_model_perdict_image():
    file = request.files['image'] # Get file from HTTP    
    filepath = UPLOADED_FILE_DIRECTORY + file.filename # Create full filepath      
    file.save(filepath) # Save file
    # # Read the image with tensorflow
    # tf_image = tf.io.read_file(filepath)
    # # Decode and convert to gray scale (whith channels = 1)
    # tf_image = tf.io.decode_png(tf_image, channels=1)
    # # Convert image array into float32 in range [0, 1]
    # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
    # # Resize image to desired size [height, width]
    # tf_image = tf.image.resize(tf_image, [50, 200])
    # # Transpose the image data array because we want the third dimension to corresponde to the width
    # tf_image = tf.transpose(tf_image, perm=[1, 0, 2])

    #########
    #########

    data = tf.data.Dataset.from_tensor_slices(
        ([filepath], ["None"])
    ).map(
        encode_single_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

    for batch in data.take(1):
        batch_images = batch["image"]
        preds = prediction_model.predict(batch_images)
        pred_text = decode_cnn_ocr_prediction_model(preds)
        print(pred_text)


    # os.remove(filepath)
    
    # prediction = cnn_ocr_prediction_model.predict(tf_image)
    # predicted_text = decode_cnn_ocr_prediction_model(prediction)
    # print(predicted_text)

    return jsonify({"text": pred_text[0]})
