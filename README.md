# Group-Emotion-Recognition

This project aims to classify a group’s perceived emotion as Positive, Neutral or Negative. The dataset being used is the [Group Affect Database 3.0](https://sites.google.com/view/emotiw2018) which contains "in the wild" photos of groups of people in various social environments.

## Instructions

This repository consists of 3 branches - 
1. `master` - contains the code used to train and test the model.
2. `webapp` - contains the webapp.
3. `android` - contains the android app.

### 1. master branch

#### 1.1 To classify an image (or a set of images) using the full pipeline

1. Fork this repository and clone the forked repository.
2. Create and activate a Python 3 virtualenv.
3. Use `pip install -r requirements.txt` to install the requirements.
4. Download the final ensemble CNN model from [here](https://drive.google.com/open?id=1dkk7K_R16fW7T0ETsaaG5lT0PZG8K7uE) and model architecture from [here](https://drive.google.com/open?id=1vAR-_QIPpAVYBWNlg6E_CJ1FGnePkW2i) and paste/replace in the cloned repository.
5. Use `python classify_image.py image_dir original_label` to classify an image as Positive, Neutral or Negative. eg - `python classify_image.py input/val/Positive/ Positive` classifies the images in the `input/val/Positive/` directory with original label as `Positive`.


### 2. webapp branch

#### 2.1 To run the web app locally

1. Fork this repository and clone the forked repository. (Ignore if already done in 1.1.)
2. Use `git checkout webapp` to switch to the webapp branch.
2. Create and activate a Python 3 virtualenv. (Ignore if already done in 1.1.)
3. Use `pip install -r requirements.txt` to install the requirements.
4. `python manage.py runserver` to start the server. Frontend can be accessed at `http://127.0.0.1:5000`

---

## The Need for Emotion Recognition

So, first of all, why do we need emotion recognition?

Emotion recognition is important -

*   To improve the user’s experience, as a customer, learner, or as a generic service user.
*   Can help improve services without the need to formally and continuously ask the user for feedback.
*   Also, using automatic emotion recognition in public safety, healthcare, or assistive technology, can significantly improve the quality of people’s lives, allowing them to live in a safer environment or reducing the impact that disabilities or other health conditions have.

## Applications of Emotion Recognition

Emotion Recognition has applications in crowd analytics, social media, marketing, event detection and summarization, public safety, human-computer interaction, digital security surveillance, street analytics, image retrieval, etc.

## The rise of Group Emotion Recognition

The problem of emotion recognition for a group of people has been less extensively studied, but it is gaining popularity due to the massive amount of data available on social networking sites containing images of groups of people participating in social events.

## Challenges facing Group Emotion Recognition

Group emotion recognition is a challenging problem due to obstructions like head and body pose variations, occlusions, variable lighting conditions, variance of actors, varied indoor and outdoor settings and image quality.

## Approach

My approach is based on the research paper "[Emotion Recognition in the Wild using Deep Neural Networks and Bayesian Classifiers](https://arxiv.org/abs/1709.03820)." So, my solution is a hybrid network that incorporates deep neural networks, and Bayesian classifiers. Deep Convolutional Neural Networks (CNNs) work from bottom to top, analysing facial expressions expressed by individual faces extracted from the image. The Bayesian network works from top to bottom, inferring the global emotion for the image, by integrating the visual features of the contents of the image obtained through a scene descriptor. In the final pipeline, the group emotion category predicted by an ensemble of CNNs in the bottom layer is passed as input to the Bayesian Network in the top layer and an overall prediction for the image is obtained.

  

![Approach](https://emotion-recognition.samanyougarg.com/static/images/method.jpg)

  

### Top-down approach

**Top-down approach** considers the scene context, such as background, environment, clothes, place, etc. It consists of the following steps –

1.  Acquiring the scene descriptors
    
2.  Setting evidences in the Bayesian Network
    
3.  Estimating the posterior distribution of the Bayesian Network
    

### Bottom-up approach

**Bottom-up approach** estimates the facial expressions of each individual in the group –

1.  Face detection
    
2.  Features pre-processing
    
3.  CNN forward pass
    

The value obtained by the bottom-up module is then used as input to the Bayesian Network in the top layer.

---
