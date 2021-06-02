from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = "./savedModel"
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, model):
    x = image.img_to_array(img)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, './uploads', secure_filename(f.filename))
        f.save(file_path)

        img = cv2.imread(file_path)
        preds = model_predict(img, model)
        expression = ["angry", "happy", "sad", "neutral"]

        result = expression[np.argmax(preds)]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

