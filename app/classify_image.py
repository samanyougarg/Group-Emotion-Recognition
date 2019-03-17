"""Module to classify an image as Positive, Negative or Neutral."""

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import numpy as np
from . import image_preprocessing
from . import cnn
from . import bayesian_network

image_label_dict = {
    "image_happy_1.jpg": ['people', 'friendship', 'fun', 'event', 'drinking', 'happy', 'picnic', 'recreation', 'smile', 'leisure'],
    "image_happy_2.jpg": ['hair', 'facial expression', 'fun', 'friendship', 'hairstyle', 'smile', 'yellow', 'event', 'human', 'laugh'],
    "image_happy_neutral_1.jpg": ['people', 'family taking photos together', 'social group', 'child', 'father', 'event','family', 'fun', 'smile', 'photography'],
    "image_happy_neutral_2.jpg": ['people', 'social group', 'fun', 'team', 'event', 'crew', 'tourism', 'uniform', 'leisure', 'smile'],
    "image_happy_neutral_3.jpg": ['people', 'tribe', 'fun', 'human', 'smile', 'community', 'child', 'happy', 'adaptation'],
    "image_happy_neutral_4.jpg": ['people', 'child', 'smile', 'community', 'youth', 'friendship', 'adaptation', 'fun', 'happy', 'event'],
    "image_neutral_sad.jpg": ['hair', 'face', 'chin', 'hairstyle', 'cool', 'forehead', 'black hair', 'fun', 'neck', 'smile'],
    "image_neutral.jpg": ['face', 'people', 'facial expression', 'child', 'smile', 'skin', 'fun', 'child model','human', 'happy'],
    "image_sad_1.jpg": ['people', 'community', 'tribe', 'adaptation', 'tradition', 'event', 'child', 'smile', 'tourism', 'turban'],
    "image_negative_1.jpg": ['military', 'military uniform', 'people', 'parade', 'uniform', 'troop', 'marching', 'army', 'military officer', 'military organization'],
    "image_negative_2.jpg": ['army', 'soldier', 'military', 'military uniform', 'troop', 'people', 'military organization', 'infantry', 'military person', 'military camouflage']
}

# function to classify an image
def classify_image(image_path, cnn_model, bayesian_model, labels_list):
    # get the image name from the image path
    image_name = image_path.split('/')[-1]
    labels = []

    # if image is from collection, get the labels from the dictionary
    if image_name in image_label_dict.keys():
        labels = image_label_dict[image_name]
    # else get the labels from the Google Vision API
    else:
        labels = image_preprocessing.detect_labels(image_path)

    # labels = ['people', 'friendship', 'fun', 'event', 'drinking', 'happy', 'picnic', 'recreation', 'smile', 'leisure']

    print("RadhaKrishna")
    print(labels)

    # preprocess the image
    image_preprocessing.preprocess(os.getcwd() + "/app/static/input/", image_path)

    # Gets the following using the CNN model -
    # i. label of the predicted emotion for the whole image
    # ii. mean cnn predictions for all the faces in the image
    # iii. cnn predictions for each individual face in the image
    # iv. a boolean that specifies whether any faces were detected in the image
    cnn_label, cnn_dict, cnn_individual_dict, faces_detected = cnn.predict_image(cnn_model, os.getcwd() + "/app/static/input/Aligned/", image_path)

    # Gets the following using the Bayesian model -
    # i. label of the predicted emotion for the whole image (using the Bayesian Network)
    # ii. label of the predicted emotion for the whole image (using the Bayesian Network + CNN as a node)
    # iii. Bayesian predictions for the whole image
    # iv. Bayesian + CNN predictions for the whole image
    bayesian_label, bayesian_cnn_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    print("Faces detected: " + str(faces_detected))
    print("CNN Label: " + str(cnn_label))
    print("Bayesian Label: " + str(bayesian_label))
    print("Bayesian + CNN Label: " + str(bayesian_cnn_label))
    
    # return 
    # i. the bayesian predictions
    # ii. cnn + bayesian predictions
    # iii. cnn predictions for the group
    # iv. cnn predictions for each detected face
    return emotion_dict, emotion_cnn_dict, cnn_dict, cnn_individual_dict