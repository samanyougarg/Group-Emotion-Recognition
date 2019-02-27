import os
import sys
import glob
import numpy as np
from . import image_preprocessing
from . import cnn
from . import bayesian_network

image_label_dict = {
    "image_happy_1.jpg": ['People', 'Friendship', 'Fun', 'Event', 'Drinking', 'Happy', 'Picnic', 'Recreation', 'Smile', 'Leisure'],
    "image_happy_2.jpg": ['Hair', 'Facial expression', 'Fun', 'Friendship', 'Hairstyle', 'Smile', 'Yellow', 'Event', 'Human', 'Laugh'],
    "image_happy_neutral_1.jpg": ['People', 'Family taking photos together', 'Social group', 'Child', 'Father', 'Event','Family', 'Fun', 'Smile', 'Photography'],
    "image_happy_neutral_2.jpg": ['People', 'Social group', 'Fun', 'Team', 'Event', 'Crew', 'Tourism', 'Uniform', 'Leisure', 'Smile'],
    "image_happy_neutral_3.jpg": ['People', 'Tribe', 'Fun', 'Human', 'Smile', 'Community', 'Child', 'Happy', 'Adaptation'],
    "image_happy_neutral_4.jpg": ['People', 'Child', 'Smile', 'Community', 'Youth', 'Friendship', 'Adaptation', 'Fun', 'Happy', 'Event'],
    "image_neutral_sad.jpg": ['Hair', 'Face', 'Chin', 'Hairstyle', 'Cool', 'Forehead', 'Black hair', 'Fun', 'Neck', 'Smile'],
    "image_neutral.jpg": ['Face', 'People', 'Facial expression', 'Child', 'Smile', 'Skin', 'Fun', 'Child model','Human', 'Happy'],
    "image_sad_1.jpg": ['People', 'Community', 'Tribe', 'Adaptation', 'Tradition', 'Event', 'Child', 'Smile', 'Tourism', 'Turban']
}

# function to classify an image
def classify_image(image_path, cnn_model, bayesian_model, labels_list):
    # get the image name from the image path
    image_name = image_path.split('/')[-1]
    labels = []

    # if image is from collection, get the labels from the dictionaru
    if image_name in image_label_dict.keys():
        labels = image_label_dict[image_name]
    # else get the labels from the Google Vision API
    else:
        labels = image_preprocessing.detect_labels(image_path)

    print("RadhaKrishna")
    print(labels)

    # preprocess the image
    image_preprocessing.preprocess(os.getcwd() + "/app/static/input/", image_path)

    # get mean cnn predictions for the faces from the image
    cnn_label, cnn_dict, cnn_individual_dict = cnn.predict_image(cnn_model, os.getcwd() + "/app/static/input/Aligned/", image_path)

    # get the bayesian and bayesian + cnn predictions for the image
    bayesian_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    print("CNN Label: " + cnn_label)
    print("Bayesian Label: " + bayesian_label)
    
    # return the bayesian, cnn and bayesian + cnn predictions
    return emotion_dict, emotion_cnn_dict, cnn_dict, cnn_individual_dict