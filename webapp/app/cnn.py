from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import cv2

classes = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def load_model():
    # read the model json file
    json_file = open('../model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load the model from json file
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("../model.h5")

    print("Loaded CNN model from disk")

    return model

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

def predict_image(model, input_path, image_path):
    # extract image name from the image path
    image_name = (image_path.split("/")[-1])[:-4]

    # list to store predictions for all faces in the image
    predictions_list = []

    for face_image in glob.glob(input_path + "*.jpg"):
        face_image_name = (face_image.split("/")[-1])[:-4]
        # if face is from the current image
        if (image_name + "_face" in face_image_name):
            # get the predicted probabilities for current face image
            predicted_probabilities = predict_face(model, face_image)
            # append to the probabilities list
            predictions_list.append(predicted_probabilities)
        
    # mean of the predicted probabilities for each face in the image
    mean_probabilities_for_image = np.mean(predictions_list, axis=0)
    print(mean_probabilities_for_image)
    emotion_dict = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    emotion_dict['Positive'] = float(round(mean_probabilities_for_image[0][2], 4))
    emotion_dict['Negative'] = float(round(mean_probabilities_for_image[0][0], 4))
    emotion_dict['Neutral'] = float(round(mean_probabilities_for_image[0][1], 4))

    # predicted class for the image i.e. class with the highest probabilities
    return classes[np.argmax(mean_probabilities_for_image)], emotion_dict
