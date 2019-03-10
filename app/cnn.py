"""Module to predict the emotions for a face image using a pre-trained CNN model."""

#!/usr/bin/env python
# coding: utf-8

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob, os
import cv2

# class mapping
classes = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
# class html mapping to be used in the frontend
classes_face = {0: '<i class="fa fa-frown-o" style="font-size: 2rem; color: #dc3545;"></i>', 1: '<i class="fa fa-meh-o" style="font-size: 2rem; color: #ffc107;"></i>', 2: '<i class="fa fa-smile-o" style="font-size: 2rem; color: #28a745;"></i>'}


# function to load a pretrained CNN model
def load_model():
    # read the model json file
    json_file = open(os.getcwd() + "/app/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load the model from json file
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.getcwd() + "/app/model.h5")

    print("Loaded CNN model from disk")

    return model


# function to predict the emotions for an individual face
def predict_face(model, image_path):
    # Load the image and resize to 64X64
    image = load_img(image_path, target_size=(64, 64))
    # Covert to a numpy array
    image = img_to_array(image)
    # Normalize it
    image = image / 255
    # Expand dimensions
    image = np.expand_dims(image, axis=0)

    # Get the predicted probabilities for each class
    pred = model.predict(image)
    # Get the class with the highest probability
    pred_digits=np.argmax(pred,axis=1)

    # print("\n")
    # print((image_path.split("/")[-1])[:-4])
    # print("Predicted probabilities: " + str(pred[0])) 
    # print("Predicted Index: " + str(pred_digits[0]))
    # print("\n")

    # return the predicted probabilities for the face
    return pred


# function to predict the emotions for the whole image
def predict_image(model, input_path, image_path):
    # extract image name from the image path
    image_name = (image_path.split("/")[-1])[:-4]

    # list to store predictions for all faces in the image
    predictions_list = []
    # dict to store predictions for individual faces in the image
    predictions_dict = {}
    # a boolean that specifies whether faces were detected in the image
    faces_detected = True

    # for each face image in the input directory
    for face_image in glob.glob(input_path + "*.jpg"):
        # extract image name from the image path
        face_image_name = (face_image.split("/")[-1])[:-4]
        # if face is from the current image
        if (image_name + "_face" in face_image_name):
            # get the predicted probabilities for current face image
            predicted_probabilities = predict_face(model, face_image)
            # append to the probabilities list
            predictions_list.append(predicted_probabilities)
            # store the face and the corresponding probabilities in the dict
            predictions_dict[face_image.split("/")[-1]] = classes_face[np.argmax(predicted_probabilities)]
    
    # if predictions list is empty
    if predictions_list == []:
        # set faces_detected = False
        faces_detected = False

    # if faces were detected in the image
    if faces_detected:
        # mean of the predicted probabilities for each face in the image
        mean_probabilities_for_image = np.mean(predictions_list, axis=0)

        # print(mean_probabilities_for_image)

        emotion_dict = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        # set the mean probability for Positive emotion for the image
        emotion_dict['Positive'] = float(round(mean_probabilities_for_image[0][2], 4))
        # set the mean probability for Negative emotion for the image
        emotion_dict['Negative'] = float(round(mean_probabilities_for_image[0][0], 4))
        # set the mean probability for Neutral emotion for the image
        emotion_dict['Neutral'] = float(round(mean_probabilities_for_image[0][1], 4))

        # Return -
        # i. label of the predicted emotion for the whole image
        # ii. mean cnn predictions for all the faces in the image
        # iii. cnn predictions for each individual face in the image
        # iv. a boolean that specifies whether any faces were detected in the image
        return classes[np.argmax(mean_probabilities_for_image)], emotion_dict, predictions_dict, faces_detected
    return None, None, None, faces_detected
