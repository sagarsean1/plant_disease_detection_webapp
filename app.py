from flask import Flask, render_template, request, flash, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from flask_cors import cross_origin
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns


UPLOAD_FOLDER = './static'
if 'static' not in os.listdir():
    os.mkdir('static')
app = Flask(__name__)
app.secret_key = "alkdfueirljdaflkjdfuaiorjlkjfdla"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@cross_origin()
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template('index.html')


@cross_origin()
@app.route('/predict', methods=['POST'])
def predict():

    if len(os.listdir('static')) > 0:
        for img in os.listdir('static'):
            os.remove('static/'+img)
    plant_diseases = ['potato_early_blight',
                      'potato_healthy',
                      'potato_late_blight',
                      'tomato_healthy',
                      'tomato_early_blight',
                      'tomato_mosaic_virus',
                      'tomato_bacterial_spot',
                      'tomato__target_spot',
                      'tomato_late_blight',
                      'tomato_septoria_leaf_spot',
                      'tomato_leaf_mold',
                      'tomato__spider_mites_two-spotted_spider_mite']

    if request.method == 'POST':

        f = request.files['file']
        if f.filename == '':
            flash("No image is uploaded, Please upload image")
            return redirect(url_for('home'))
        print(f.filename)
        if not is_allowed_file(f.filename):
            flash("File format is not support try uploading .jpg or .jpeg or .png")
            return redirect(url_for('home'))
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg'))
        model = load_model('saved_model/plant_vgg19_model_weights.h5')
        outcome = model.predict(prepare('./static/img.jpg'))[0]
        data = pd.Series(outcome, index=plant_diseases)
        predict = np.argmax(outcome)
        disease = plant_diseases[predict]
        image_name = disease+'.png'
        plt.figure(figsize=(13, 9))
        sns.barplot(x=data.values, y=data.index)
        plt.title('Predicted probability % of diseases')
        plt.savefig('static/'+image_name)

    return render_template('index.html', result=disease, image=image_name)


def prepare(image):
    im = cv2.imread(image)
    im = cv2.resize(im, (256, 256)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)
    return im


def is_allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext


if __name__ == "__main__":

    app.run(debug=True)
