"""Module to evaluate full pipeline on the validation set.

    python evaluate.py
"""

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import numpy as np
import image_preprocessing
import cnn
import bayesian_network
import json
import pandas as pd

# class mapping
classes = {"Positive": 0, "Neutral": 1, "Negative": 2, "None": 3}

# function to classify an image
def classify_image(image_folder_path, image_name, real_label, cnn_model, bayesian_model, labels_list):
    with open('val_labels.json', mode='r', encoding='utf-8') as f:
        image_labels_dict = json.load(f)
    labels = image_labels_dict[image_name]

    # print("RadhaKrishna")
#     print(labels)

    # preprocess the image
    image_preprocessing.preprocess(image_folder_path, image_name)

    # get mean cnn predictions for the faces from the image
    cnn_label, cnn_dict, faces_detected = cnn.predict_image(cnn_model, image_folder_path + "Aligned/", image_name)

    # get the bayesian and bayesian + cnn predictions for the image
    bayesian_label, bayesian_cnn_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    # print("Faces detected: " + str(faces_detected))
    # print("Real Label: " + str(real_label))
    # print("CNN Label: " + str(cnn_label))
    # print("Bayesian Label: " + str(bayesian_label))
    # print("Bayesian + CNN Label: " + str(bayesian_cnn_label))

    return classes[real_label], classes[str(cnn_label)], classes[str(bayesian_label)], classes[str(bayesian_cnn_label)], faces_detected



# load the cnn model
cnn_model = cnn.load_model()
# load the bayesian model
bayesian_model, labels_list = bayesian_network.load_model()


# function to evaluate the pipeline on a given directory
def evaluate(image_folder_path, real_label):
    # print("RadhaKrishna")
    # get the count of total number of files in the directory
    _, _, files = next(os.walk(image_folder_path))
    file_count = len(files)-1
    # list to store the predictions
    predictions = []
    # set count = 1
    i = 1

    # for each image in the directory
    for file in sorted(glob.glob(image_folder_path + "*.jpg")):
        # extract the image name
        image_name = (file.split('/'))[-1]
        print("Image: " + image_name)
        print(str(i) + "/" + str(file_count))
        # create a dict to store the image name and predictions
        prediction = {"Image": image_name}
        prediction["Actual"], prediction["CNN"], prediction["Bayesian"], prediction["Bayesian + CNN"], prediction["Faces Detected"] = classify_image(image_folder_path, image_name, real_label, cnn_model, bayesian_model, labels_list)
        # append the dict to the list of predictions
        predictions.append(prediction)
        # increase the count
        i+=1
    # return the predictions list
    return predictions

# class list
class_list = ['Positive', 'Neutral', 'Negative']
predictions_list = []

# for each class in the class list
for emotion_class in class_list:
    # evaluate all the images in that folder
    predictions = evaluate('input/val/' + emotion_class + '/', emotion_class)
    # add the predictions to the predictions list
    predictions_list += predictions
# create a pandas dataframe from the predictions list
df = pd.DataFrame(predictions_list)
# store the dataframe to a file
df.to_pickle('predictions')




