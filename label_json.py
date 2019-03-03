import os
import sys
import glob
import image_preprocessing
import json

def label_image(image_path, image_name, real_label, filename):
    # labels = image_preprocessing.detect_labels(image_path + image_name)
    labels = ['people', 'friendship', 'fun', 'event', 'drinking', 'happy', 'picnic', 'recreation', 'smile', 'leisure']

    print("RadhaKrishna")
    print("Image name: " + image_name)
    print(labels)

    with open(filename, mode='r+', encoding='utf-8') as f:
        first = f.read(1)
        if not first:
            json.dump({}, f)

    with open(filename, mode='r', encoding='utf-8') as f:
        image_labels_dict = json.load(f)
    
    with open(filename, mode='w', encoding='utf-8') as f:
        image_labels_dict[image_name] = labels
        json.dump(image_labels_dict, f)


def main(image_path, real_label, filename):
    print("RadhaKrishna")
    for file in sorted(glob.glob(image_path + "*.jpg")):
        image_name = (file.split('/'))[-1]
        print("Image: " + image_name)
        label_image(image_path, image_name, real_label, filename)

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])