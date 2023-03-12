from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

app = Flask(__name__)

UPLOADED_FILE_DIRECTORY = os.getcwd() + "/" + "__uploaded_files__/"
# Create uploaded file directory if does not exists
if os.path.exists(UPLOADED_FILE_DIRECTORY) == False:
    os.makedirs(UPLOADED_FILE_DIRECTORY)

cnn_ocr_model = tf.keras.models.load_model('./models/CNN-OCR-MODEL-SAVE-V2/')
cnn_ocr_prediction_model = keras.models.Model(
    cnn_ocr_model.get_layer(name='image').input,
    cnn_ocr_model.get_layer(name='dense2').output
)

@app.route('/ping')
def ping():
    return "AI API running"

#!!!!!!temp
max_length = 50

characters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def decode_cnn_ocr_prediction_model(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # Iterate over the result and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace('[UNK]', '').strip('0')
        output_text.append(res)
    return output_text

image_height = 32
image_width = 128

def encode_single_sample(image_path, label):
    # Read the image with tensorflow
    image = tf.io.read_file(image_path)
    # Decode and convert to gray scale (whith channel = 1), we don't need colors to get the label of an image
    image = tf.io.decode_png(image, channels=1)
    # Convert image array into float32 in range [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize image to the desired size
    image = tf.image.resize(image, [image_height, image_width])
    # Transpose the image data array because we want the third dimension to corresponde to the width
    image = tf.transpose(image, perm=[1, 0, 2])
    # Map the label characters to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return the corresponding dictionary
    return {"image": image, "label": label}

@app.route('/cnn-ocr-model/predict_image', methods=['POST'])
def ocr_captcha_model_perdict_image():
    # Get file from HTTP
    file = request.files['image']
    # Create full filepath  
    filepath = UPLOADED_FILE_DIRECTORY + file.filename
    # Save file
    file.save(filepath)
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
        preds = cnn_ocr_prediction_model.predict(batch_images)
        pred_text = decode_cnn_ocr_prediction_model(preds)
        print(pred_text)


    # os.remove(filepath)
    
    # prediction = cnn_ocr_prediction_model.predict(tf_image)
    # predicted_text = decode_cnn_ocr_prediction_model(prediction)
    # print(predicted_text)

    return jsonify({"text": pred_text[0]})
