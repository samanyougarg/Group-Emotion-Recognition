"""Module to predict the emotions for a face image using a pre-trained CNN model."""

#!/usr/bin/env python
# coding: utf-8

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob, os
import cv2
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
from keras.engine import training
from tensorflow.python.framework.ops import Tensor
from keras.models import Model, Input
from keras.layers import Average
import keras.layers as layers

# class mapping
classes = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# function to create an ensemble of models
def ensemble_models(models, model_input):
    # append the output of each model to a list
    model_output = [model(model_input) for model in models] 
    # average the outputs of each model
    avg_output = layers.average(model_output) 
    # build the ensemble model having same input as our models but the average
    # of the output of our models as output
    ensemble_model = Model(inputs=model_input, outputs=avg_output, name='ensemble')  
    # return the ensembled model
    return ensemble_model

# function to load a pretrained CNN models
def load_model():
    # Generate an ensemble of models and save to disk

    # model_names = ["model_vgg", "model_aligned_gen_128", "model_vggface", "model_aligned_gen_128_2048"]
    # models = []

    # for model_name in model_names:
    #     # read the model json file
    #     json_file = open('models/' + model_name + '.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()

    #     # load the model from json file
    #     model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     model.load_weights('models/' + model_name + '.h5')
        
    #     # append the model to our models list
    #     models.append(model)

    # # input layer for our ensemble model
    # model_input = Input(shape = models[0].input_shape[1:])
    # # create the ensemble model from our models
    # ensemble_model = ensemble_models(models, model_input)

    # # serialize model to JSON
    # model_json = ensemble_model.to_json()
    # with open("radhakrishna.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # ensemble_model.save_weights("radhakrishna.h5")
    # print("Saved model to disk")

    # # -------------------------------------------------------------------

    # Load the saved ensemble model

    # read the model json file
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load the model from json file
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('model.h5')

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
        # iii. a boolean that specifies whether any faces were detected in the image
        return classes[np.argmax(mean_probabilities_for_image)], emotion_dict, faces_detected
    return None, None, faces_detected
