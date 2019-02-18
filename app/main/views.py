#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import render_template, jsonify, current_app, request
from app.models.building import BuildingModel
from . import main
from app import db, csrf
from werkzeug.utils import secure_filename
from .forms import ImageForm
import os
from app import cnn
from app import bayesian_network
from app.classify_image import classify_image
import tensorflow as tf 


def load_models():
    global cnn_model, graph, bayesian_model, labels_list
    cnn_model = cnn.load_model()
    cnn_model._make_predict_function()
    graph = tf.get_default_graph()
    bayesian_model, labels_list = bayesian_network.load_model()

@main.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    return render_template('main/index.html', form=form)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in set(['png', 'jpg', 'jpeg', 'gif'])


@main.route('/process-image/', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':

        if 'image' in request.files:
            image = request.files['image']
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                current_app.logger.info('FileName: ' + filename)
                updir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static/input'))
                image_path = os.path.join(updir, filename)
                image.save(image_path)

        elif request.form['image_set'] is not None:
            filename = (request.form['image_set']).split('/')[-1]
            updir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static/input'))
            image_path = os.path.join(updir, filename)

        emotion_dict, emotion_cnn_dict, cnn_dict = classify_image(image_path, cnn_model, graph, bayesian_model, labels_list)
        
        return jsonify(emotion_dict, cnn_dict, emotion_cnn_dict)