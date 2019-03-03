import os
import sys
import glob
import numpy as np
import image_preprocessing
import cnn
import bayesian_network
import json

image_label_dict = {
    "image_happy_1.jpg": ['people', 'friendship', 'fun', 'event', 'drinking', 'happy', 'picnic', 'recreation', 'smile', 'leisure'],
    "image_happy_2.jpg": ['hair', 'facial expression', 'fun', 'friendship', 'hairstyle', 'smile', 'yellow', 'event', 'human', 'laugh'],
    "image_happy_neutral_1.jpg": ['people', 'family taking photos together', 'social group', 'child', 'father', 'event','family', 'fun', 'smile', 'photography'],
    "image_happy_neutral_2.jpg": ['people', 'social group', 'fun', 'team', 'event', 'crew', 'tourism', 'uniform', 'leisure', 'smile'],
    "image_happy_neutral_3.jpg": ['people', 'tribe', 'fun', 'human', 'smile', 'community', 'child', 'happy', 'adaptation'],
    "image_happy_neutral_4.jpg": ['people', 'child', 'smile', 'community', 'youth', 'friendship', 'adaptation', 'fun', 'happy', 'event'],
    "image_neutral_sad.jpg": ['hair', 'face', 'chin', 'hairstyle', 'cool', 'forehead', 'black hair', 'fun', 'neck', 'smile'],
    "image_neutral.jpg": ['face', 'people', 'facial expression', 'child', 'smile', 'skin', 'fun', 'child model','human', 'happy'],
    "image_sad_1.jpg": ['people', 'community', 'tribe', 'adaptation', 'tradition', 'event', 'child', 'smile', 'tourism', 'turban']
}


def classify_image(image_name, real_label, cnn_model, bayesian_model, labels_list):
    # if image is from collection, get the labels from the dictionaru
#     if image_name in image_label_dict.keys():
#         print("RadhaKrishna")
#         labels = image_label_dict[image_name]
#     # else get the labels from the Google Vision API
#     else:
#         labels = image_preprocessing.detect_labels("input/" + image_name)

    labels = ['people', 'friendship', 'fun', 'event', 'drinking', 'happy', 'picnic', 'recreation', 'smile', 'leisure']

    print("RadhaKrishna")
    print(labels)

    # preprocess the image
    image_preprocessing.preprocess("input/", image_name)

    # get mean cnn predictions for the faces from the image
    cnn_label, cnn_dict, faces_detected = cnn.predict_image(cnn_model, "input/Aligned/", image_name)

    # get the bayesian and bayesian + cnn predictions for the image
    bayesian_label, bayesian_cnn_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    print("Faces detected: " + str(faces_detected))
    print("Real Label: " + str(real_label))
    print("CNN Label: " + str(cnn_label))
    print("Bayesian Label: " + str(bayesian_label))
    print("Bayesian + CNN Label: " + str(bayesian_cnn_label))

    return bayesian_cnn_label

def main(test_path, real_label):
    print("RadhaKrishna")
    cnn_model = cnn.load_model()
    bayesian_model, labels_list = bayesian_network.load_model()
    for file in sorted(glob.glob(test_path + "*.jpg")):
        image_name = (file.split('/'))[-1]
        print("Image: " + image_name)
        prediction = classify_image(image_name, real_label, cnn_model, bayesian_model, labels_list)
        break


if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])