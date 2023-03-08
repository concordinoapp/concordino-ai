from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

app = Flask(__name__)

# basic_ocr_model = tf.keras.models.load_model('./models/basic-ocr-model')
ocr_captcha_model = tf.keras.models.load_model('./models/ocr-captcha-model')
ocr_captcha_prediction_model = keras.models.Model(
    ocr_captcha_model.get_layer(name='image').input,
    ocr_captcha_model.get_layer(name='dense2').output
)
characters = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


@app.route('/ping')
def ping():
    return "AI API running"

# @app.route("/basic-ocr-model/predict_image", methods=['POST'])
# def basic_ocr_model_predict_image():
#     file = request.files['image']
#     # load image and convert it to gray scale and resize it to 28*28 pixel image
#     img = Image.open(file.stream).convert('L').resize((28, 28))
#     # convert to 2D array and divide by 255.0 to get float values between 0 and 1
#     img_array = np.array(img) / 255.0
#     predictions = basic_ocr_model.predict( np.array([img_array]) )
#     result = np.where(predictions[0] == np.amax(predictions[0]))[0][0]    
#     return jsonify({'found number': int(result)})


# A utility function to decode the output of the captcha network
def catpcha_model_decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # Iterate over the result and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

@app.route('/ocr-captcha-model/predict_image', methods=['POST'])
def ocr_captcha_model_perdict_image():
    file = request.files['image']
    # load image and convert it to gray scale and resize it to 200*50 pixel image
    img = Image.open(file.stream).convert('L').resize((200, 50))
    # convert to 2D array and divide by 255.0 to get float values between 0 and 1
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((200, 50, 1))
    print(f"img_array shape = {img_array.shape}")
    pred = ocr_captcha_prediction_model.predict(img_array)
    print(pred)
    return "OK"