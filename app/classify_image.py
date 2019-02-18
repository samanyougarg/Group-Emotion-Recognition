import os
import sys
import glob
import numpy as np
from . import image_preprocessing
from . import cnn
from . import bayesian_network
from flask import current_app

def classify_image(image_path, cnn_model, bayesian_model, labels_list):
    # labels = image_preprocessing.detect_labels(image_path)
    labels = ["people", "tribe", "leisure", "fun", "temple", "vacation", "tradition", "happiness", "tourism", "travel"]

    image_preprocessing.preprocess(os.getcwd() + "/app/static/input/", image_path)

    cnn_label, cnn_dict = cnn.predict_image(cnn_model, os.getcwd() + "/app/static/input/Aligned/", image_path)

    bayesian_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    prediction = bayesian_label

    current_app.logger.info("CNN Label: " + cnn_label)
    current_app.logger.info("Bayesian Label: " + bayesian_label)
    
    return emotion_dict, emotion_cnn_dict, cnn_dict