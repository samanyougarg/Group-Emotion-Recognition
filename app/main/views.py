#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import render_template, jsonify, current_app, request, send_from_directory
from app.models.building import BuildingModel
from . import main
from app import db, csrf
from werkzeug.utils import secure_filename
from .forms import ImageForm
import os
from app import cnn
from app import bayesian_network
from app.classify_image import classify_image
from PIL import Image

size = 1000, 1000

# function to load the cnn and bayesian models
def load_models():
    global cnn_model, bayesian_model, labels_list
    cnn_model = cnn.load_model()
    cnn_model._make_predict_function()
    bayesian_model, labels_list = bayesian_network.load_model()

# main route
@main.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    return render_template('main/index.html', form=form)


# function to check if the uploaded file is an image
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in set(['png', 'jpg', 'jpeg', 'gif'])


# function to process the selected image
@main.route('/process-image/', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        # if the user uploaded an image
        if 'image' in request.files:
            # get the image
            image = request.files['image']
            # if the image is of correct type
            if image and allowed_file(image.filename):
                # get the name of the image
                filename = secure_filename(image.filename)
                # print the image name
                current_app.logger.info('FileName: ' + filename)
                updir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static/input'))
                image_path = os.path.join(updir, filename)
                # save the image
                image.save(image_path)

        # else if user selected an image
        elif request.form['image_set'] is not None:
            # get the name of the image
            filename = (request.form['image_set']).split('/')[-1]
            # print the image name
            current_app.logger.info('FileName: ' + filename)
            updir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static/input'))
            image_path = os.path.join(updir, filename)

        # read the image
        im = Image.open(image_path)
        # resize the image
        im.thumbnail(size, Image.ANTIALIAS)
        # save
        im.save(image_path, "JPEG")

        # get the bayesian, cnn and bayesian + cnn predictions
        emotion_dict, emotion_cnn_dict, cnn_dict, cnn_individual_dict = classify_image(image_path, cnn_model, bayesian_model, labels_list)
        
        # return bayesian, cnn and bayesian + cnn predictions
        return jsonify(emotion_dict, cnn_dict, emotion_cnn_dict, cnn_individual_dict)


@main.route('/robots.txt')
@main.route('/sitemap.xml')
@main.route('/manifest.json')
@main.route('/radhakrishna.js')
def static_from_root():
    return send_from_directory(current_app.static_folder, request.path[1:])