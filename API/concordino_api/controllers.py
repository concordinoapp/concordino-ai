import os
from flask import request, jsonify
import tensorflow as tf
from .encoding import decode_cnn_ocr_prediction_model, encode_single_sample

def ping():
    return "AI API running"

def ocr_model_perdict_image(pred_model, uploaded_file_dir, char_to_num, num_to_char):
    file = request.files['image'] # Get file from HTTP    
    filepath = uploaded_file_dir + file.filename # Create full filepath      
    file.save(filepath) # Save file

    data = tf.data.Dataset.from_tensor_slices(
        ([filepath], ["None"])
    ).map(
        lambda image_path, label: encode_single_sample(image_path, label, char_to_num),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    for batch in data.take(1):
        batch_images = batch["image"]
        preds = pred_model.predict(batch_images)
        pred_text = decode_cnn_ocr_prediction_model(preds, num_to_char)
    os.remove(filepath)        
    return jsonify({"text": pred_text[0]})
