from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'Model/VGG.h5'
print('Model loading has been started....')
# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    
    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)

    if(pred[0][0]==1):
        result = "COVID-19 Positive (+ve)"
    else:
        result = "COVID-19 Negative (-ve)"

    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        out = model_predict(file_path, model)
        print(out)
        if os.path.exists(file_path):
            os.remove(file_path)

        return out
    return None


if __name__ == '__main__':
    app.run(debug=True)

