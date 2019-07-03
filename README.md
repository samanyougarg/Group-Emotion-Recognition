# Group-Emotion-Recognition

![Python](https://camo.githubusercontent.com/c589348df8bb82948f724198f52725d3d36ce738/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e782d627269676874677265656e2e737667)

This project aims to classify a group’s perceived emotion as Positive, Neutral or Negative. The dataset being used is the [Group Affect Database 3.0](https://sites.google.com/view/emotiw2018) which contains "in the wild" photos of groups of people in various social environments.

Our solution is a hybrid machine learning system that builds on the model by [Surace et al.](https://arxiv.org/abs/1709.03820) and extends it further with additional and more refined machine learning methods and experiments. It has been published in the paper [Group Emotion Recognition Using Machine Learning](https://arxiv.org/pdf/1905.01118.pdf).


## Demo

### 1. Video
[![Youtube Video](http://i.imgur.com/GPgEKL0.png)](https://youtu.be/fQrRjKQeAhM "Youtube Video")

### 2. Apps

[<img src="https://play.google.com/intl/en_us/badges/images/generic/en_badge_web_generic.png"
      alt="Download from Google Play"
      height="81">](https://play.google.com/store/apps/details?id=com.hanuman.groupemotionrecognition)
[<img src="https://i.imgur.com/6B3Qw5s.png"
      alt="Web App"
      height="80">](http://emotion-recognition.samanyougarg.com)

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

Our solution is a pipeline based approach which integrates two modules (that work in parallel): bottom-up and top-down modules, based on the idea that the emotion of a group of people can be deduced using both bottom-up and top-down approaches.

- The bottom-up module detects and extracts individual faces present in the
image and passes them as input to an ensemble of pre-trained Deep
Convolutional Neural Networks (CNNs).
- Simultaneously, the top-down module detects the labels associated with the
scene and passes them as input to a Bayesian Network (BN) which predicts
the probabilities of each class.
- In the final pipeline, the group emotion category predicted by the bottom-up
module is passed as input to the Bayesian Network in the top-down module
and an overall prediction for the image is obtained.

An overview of the full pipeline is shown in the figure below.
![Approach](https://emotion-recognition.samanyougarg.com/static/images/method.jpg)
