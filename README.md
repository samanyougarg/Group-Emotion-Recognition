# Group-Emotion-Recognition

This project aims to classify a group’s perceived emotion as Positive, Neutral or Negative. The dataset being used is the [Group Affect Database 3.0](https://sites.google.com/view/emotiw2018) which contains "in the wild" photos of groups of people in various social environments.

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

My approach is based on the research paper "[Emotion Recognition in the Wild using Deep Neural Networks and Bayesian Classifiers](https://arxiv.org/abs/1709.03820)." So, the model is basically a novel combination of deep neural networks and Bayesian classifiers. The neural network works from the bottom to the top, analysing emotions expressed by isolated faces. The Bayesian classifier estimates a global emotion integrating top-down features obtained through a scene descriptor.

  

![Approach](https://emotion-recognition.samanyougarg.com/static/images/method.jpg)

  

### Top-down approach

**Top-down approach** considers the scene context, such as background, clothes, place, etc. It consists of the following steps –

1.  Acquiring the scene descriptors
    
2.  Setting evidences in the Bayesian Network
    
3.  Estimating the posterior distribution of the Bayesian Network
    

### Bottom-up approach

**Bottom-up approach** estimates the facial expressions of each person in the group –

1.  Face detection
    
2.  Features pre-processing
    
3.  CNN forward pass
    

The value obtained by the bottom-up module is then used as input to the Bayesian Network in the top layer.

---

राधे राधे 🕉️🕉️
