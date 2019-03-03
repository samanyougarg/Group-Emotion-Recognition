import os
import sys
import glob
import numpy as np
import image_preprocessing
import cnn
import bayesian_network
import json
import pandas as pd

classes = {"Positive": 0, "Neutral": 1, "Negative": 2, "None": 3}

def classify_image(image_folder_path, image_name, real_label, cnn_model, bayesian_model, labels_list):
    with open('val_labels_positive.json', mode='r', encoding='utf-8') as f:
        image_labels_dict = json.load(f)
    labels = image_labels_dict[image_name]

    print("RadhaKrishna")
    print(labels)

    # preprocess the image
    image_preprocessing.preprocess(image_folder_path, image_name)

    # get mean cnn predictions for the faces from the image
    cnn_label, cnn_dict, faces_detected = cnn.predict_image(cnn_model, "input/Aligned/", image_name)

    # get the bayesian and bayesian + cnn predictions for the image
    bayesian_label, bayesian_cnn_label, emotion_dict, emotion_cnn_dict = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    print("Faces detected: " + str(faces_detected))
    print("Real Label: " + str(real_label))
    print("CNN Label: " + str(cnn_label))
    print("Bayesian Label: " + str(bayesian_label))
    print("Bayesian + CNN Label: " + str(bayesian_cnn_label))

    return classes[real_label], classes[cnn_label], classes[bayesian_label], classes[bayesian_cnn_label], faces_detected

def main(image_folder_path, real_label):
    print("RadhaKrishna")
    cnn_model = cnn.load_model()
    bayesian_model, labels_list = bayesian_network.load_model()
    predictions_list = []
    for file in sorted(glob.glob(image_folder_path + "*.jpg")):
        image_name = (file.split('/'))[-1]
        print("Image: " + image_name)
        prediction = {"Image": image_name}
        prediction["Actual"], prediction["CNN"], prediction["Bayesian"], prediction["Bayesian + CNN"], prediction["Faces Detected"] = classify_image(image_folder_path, image_name, real_label, cnn_model, bayesian_model, labels_list)
        predictions_list.append(prediction)
        break
    df = pd.DataFrame(predictions_list)
    print(df.head())


if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])