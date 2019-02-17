#!/usr/bin/env python
# coding: utf-8

import cv2
import os
from PIL import Image
import glob
import numpy as np
from AlignDlib import AlignDlib
import io

# ## Face and Label Detection

# ### 1. Using Google Vision API

# #### 1.1 Import Google Vision Library
from google.cloud import vision


# #### 1.2 Label Detection function
def detect_labels(path):
    """Detects labels in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    labels_list = [label.description for label in labels]
    
    return labels_list


# #### 1.3 Face Detection function
def detect_face_google(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = vision.types.Image(content=content)

    return client.face_detection(image=image).face_annotations


# #### 1.4 Function to crop the detected faces
def crop_faces_google(image_file, cropped_images_path, faces):
    count = 1
    # open the image
    image = Image.open(image_file)
    
#     print(len(faces))

    # if faces is null, then it means no face was detected in the image
    if not faces:
        print("No face detected in the image.")
        return
    
    # for each detected face in the faces list
    for face in faces:
        # get the coordinates for each vertex
        coordinates = [(vertex.x, vertex.y)
            for vertex in face.bounding_poly.vertices]
        
        # separate the x and y coordinates
        x_coordinates, y_coordinates = [], []
        for vertex in face.bounding_poly.vertices:
            x_coordinates.append(vertex.x)
            y_coordinates.append(vertex.y)
        
        x0, x1, y0, y1 = x_coordinates[0], x_coordinates[2], y_coordinates[0], y_coordinates[2]
        
        # set the coordinates of the box for each face
        box = (x0, y0, x1, y1)
        # crop the image using coordinates of the box
        cropped_image = image.crop(box)
        
        # extract image name from filename
        image_name = (image_file.split("/")[-1])[:-4]
        # save the cropped image
        cropped_image.save(cropped_images_path + image_name + "_face_" + str(count) + ".jpg")
        count+=1


# #### 1.5 Function to resize the cropped faces
def resize_faces_google(cropped_images_path, scaled_images_path, size):
    count = 1
    # for each image in the cropped images path
    for file in glob.glob(cropped_images_path+"*.jpg"):
        # read the image
        image = cv2.imread(file)
        # get the height and width of the image
        height, width = image.shape[:2]
        
        # get the height and weight ratios
        height_ratio, width_ratio = float(size/height), float(size/width)

        # resize the image making sure that the original ratio is maintained
        resized = cv2.resize(image, None, fx=width_ratio, fy=height_ratio, interpolation=cv2.INTER_AREA)
        
        # extract image name from full file name
        image_name = (file.split("/")[-1])
        # save the scaled image
        cv2.imwrite(scaled_images_path + image_name, resized)


# #### 1.6 Apply preprocessing to the dataset using the functions above
def preprocess_google(image_file):
    with open(image_file, 'rb') as image:
        # detect faces in the image
        faces = detect_face_google(image)
        
        # Reset file pointer, so we can read the file again
        image.seek(0)
        # crop detected faces and save in "Faces" directory
        crop_faces_google(image_file, "test/Faces/", faces)

    # resize the cropped faces stored in the "Faces" directory and save in "Scaled" directory
    resize_faces_google("test/Faces/", "test/Scaled/", 64)

# ---

# ### 2. Using OpenCV DNN

# #### 2.1 Load the serialized model from disk
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt',
                               'res10_300x300_ssd_iter_140000.caffemodel'
                               )


# #### 2.2 Function to detect and crop faces
def extract_faces_cv(image_file, cropped_images_path):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(image_file)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300,
                                 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    faces = net.forward()
    count = 1
    image = Image.open(image_file)
    
    if faces is None:
        print("No face detected in the image.")
        return

    # loop over the faces
    for i in range(faces.shape[2]):

        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = faces[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            cropped_image = image.crop(box)

            image_name = (image_file.split("/")[-1])[:-4]
            cropped_image.save(cropped_images_path + image_name + "_face_" + str(count) + ".jpg")
            count+=1


# #### 2.3 Function to resize the cropped faces
def resize_faces_cv(cropped_images_path, scaled_images_path, size):
    count = 1
    # for each image in the cropped images path
    for file in glob.glob(cropped_images_path+"*.jpg"):
        # read the image
        image = cv2.imread(file)
        # get the height and width of the image
        height, width = image.shape[:2]
        
        # get the height and weight ratios
        height_ratio, width_ratio = float(size/height), float(size/width)

        # resize the image making sure that the original ratio is maintained
        resized = cv2.resize(image, None, fx=width_ratio, fy=height_ratio, interpolation=cv2.INTER_AREA)
        
        # extract image name from full file name
        image_name = (file.split("/")[-1])
        # save the scaled image
        cv2.imwrite(scaled_images_path + image_name, resized)

def preprocess_cv(image_file):
    # detect and crop faces in the image
    extract_faces_cv(image_file, "test/Faces/")
    # resize the cropped faces and save in "Scaled" directory
    resize_faces_cv("test/Faces/", "test/Scaled/", 64)

# ---

# ### 3. Using Dlib

# #### 3.1 Import Dlib
import dlib

# #### 3.2 Function to detect and crop faces
# dlib hog + svm based face detector
detector = dlib.get_frontal_face_detector()

def extract_faces(image_file, cropped_images_path):
    # load input image
    image = cv2.imread(image_file)
    count = 1
    # get the image height and width
    image_height, image_width = image.shape[:2]
    
    if image is None:
        print("Could not read input image")
        exit()
    
    # apply face detection
    faces = detector(image, 1)
    
    # loop over detected faces
    for face in faces:
        # crop the image
        cropped_image = image[max(0, face.top()): min(face.bottom(), image_height),
                    max(0, face.left()): min(face.right(), image_width)]
        # extract image name from filename
        image_name = (image_file.split("/")[-1])[:-4]
        # save the cropped image
        cv2.imwrite(cropped_images_path + image_name + "_face_" + str(count) + ".jpg", cropped_image)
        count+=1


# #### 3.3 Function to resize the cropped faces
def resize_faces(cropped_images_path, scaled_images_path, size):
    count = 1
    # for each image in the cropped images path
    for file in glob.glob(cropped_images_path+"*.jpg"):
        # read the image
        image = cv2.imread(file)
        # get the height and width of the image
        height, width = image.shape[:2]
        
        # get the height and weight ratios
        height_ratio, width_ratio = float(size/height), float(size/width)
        
        # resize the image making sure that the original ratio is maintained
        resized = cv2.resize(image, None, fx=width_ratio, fy=height_ratio, interpolation=cv2.INTER_AREA)
        
        # extract image name from full file name
        image_name = (file.split("/")[-1])
        # save the scaled image
        cv2.imwrite(scaled_images_path + image_name, resized)

# #### 3.5 Function to align the faces
align_dlib = AlignDlib('shape_predictor_68_face_landmarks.dat')

def align_faces(scaled_images_path, aligned_images_path):
    count = 1
    # for each image in the scaled images directory
    for file in glob.glob(scaled_images_path+"*.jpg"):
        
        # print(file)
        
        # read the image
        image = cv2.imread(file)
        
        # initialize the bounding box
        bb = align_dlib.getLargestFaceBoundingBox(image)
        # align the face
        aligned = align_dlib.align(64, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        # if aligned
        if aligned is not None:
            image_name = (file.split("/")[-1])
            # save the image in the aligned images directory
            cv2.imwrite(aligned_images_path + image_name, aligned)
        else:
            # save the image without alignment in the aligned images directory
            cv2.imwrite(aligned_images_path + image_name, image)


# #### 3.6 Apply preprocessing to the dataset using the functions above
def preprocess(image_file):
    # detect and crop faces in the image
    extract_faces(image_file, "test/Faces/")
    # resize the cropped faces and save in "Scaled" directory
    resize_faces("test/Faces/", "test/Scaled/", 64)
    # align the scaled faces and save in "Aligned" directory
    align_faces("test/Scaled/", "test/Aligned/")



