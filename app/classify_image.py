import os
import sys
import glob
import numpy as np
from . import image_preprocessing
from . import cnn
from . import bayesian_network

# function to classify an image
def classify_image(image_path, cnn_model, bayesian_model, labels_list):
    # get the labels for the image from Google Vision API
    # labels = image_preprocessing.detect_labels(image_path)
    labels = ["people", "tribe", "leisure", "fun", "temple", "vacation", "tradition", "happiness", "tourism", "travel"]

    print("RadhaKrishna")
    print(labels)

    # preprocess the image
    image_preprocessing.preprocess(os.getcwd() + "/app/static/input/", image_path)

    # get mean cnn predictions for the faces from the image
    cnn_label, cnn_dict = cnn.predict_image(cnn_model, os.getcwd() + "/app/static/input/Aligned/", image_path)

    # get the bayesian and bayesian + cnn predictions for the image
    bayesian_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    print("CNN Label: " + cnn_label)
    print("Bayesian Label: " + bayesian_label)
    
    # return the bayesian, cnn and bayesian + cnn predictions
    return emotion_dict, emotion_cnn_dict, cnn_dict