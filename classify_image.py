import os
import sys
import glob
import numpy as np
import image_preprocessing
import cnn
import bayesian_network

def classify_image(image_path, real_label, cnn_model, bayesian_model, labels_list):
    # labels = image_preprocessing.detect_labels(image_path)
    labels = ["people", "tribe", "leisure", "fun", "temple", "vacation", "tradition", "happiness", "tourism", "travel"]

    image_preprocessing.preprocess(image_path)

    cnn_label = cnn.predict_image(cnn_model, "test/Aligned/", image_path)

    bayesian_label = bayesian_network.inference(bayesian_model, labels_list, labels, cnn_label)

    prediction = bayesian_label

    print("Real Label: " + real_label)
    print("Bayesian Label: " + bayesian_label)
    print("CNN Label: " + cnn_label)

    return prediction

def main(test_path, real_label):
    print("RadhaKrishna")
    cnn_model = cnn.load_model()
    bayesian_model, labels_list = bayesian_network.load_model()
    predicted_counter = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for image_path in sorted(glob.glob(test_path + "*.jpg")):
        print("Image: " + image_path)
        prediction = classify_image(image_path, real_label, cnn_model, bayesian_model, labels_list)
        predicted_counter[prediction] += 1
        print(predicted_counter)
        break


if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])