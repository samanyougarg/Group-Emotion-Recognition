"""Module to evaluate full pipeline on the validation set.

    python label_json.py input/val/Positive/ Positive val_labels_positive.json

"""

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import image_preprocessing
import json

# function to get the labels for an image and store to the json file
def label_image(image_path, image_name, real_label, filename):
    # detect labels for the image
    labels = image_preprocessing.detect_labels(image_path + image_name)

    # labels = ['people', 'friendship', 'fun', 'event', 'drinking', 'happy', 'picnic', 'recreation', 'smile', 'leisure']

    print("RadhaKrishna")
    print("Image name: " + image_name)
    print(labels)

    # open the file
    with open(filename, mode='r+', encoding='utf-8') as f:
        # read the first byte
        first = f.read(1)
        # if file is empty, dump an empty dict to the file
        if not first:
            json.dump({}, f)

    # open the file again
    with open(filename, mode='r', encoding='utf-8') as f:
        # load the contents of the file as a dict
        image_labels_dict = json.load(f)
    
    # open the file in writable mode
    with open(filename, mode='w', encoding='utf-8') as f:
        # create a dict with image name as key and its labels as value
        image_labels_dict[image_name] = labels
        # dump the dict to the file
        json.dump(image_labels_dict, f)


def main(image_path, real_label, filename):
    print("RadhaKrishna")
    # for each image in the image path
    for file in sorted(glob.glob(image_path + "*.jpg")):
        # extract the image name
        image_name = (file.split('/'))[-1]
        print("Image: " + image_name)
        # get the labels for the image and store to the json file
        label_image(image_path, image_name, real_label, filename)

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])