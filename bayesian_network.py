"""Module to predict the emotions for a group image using a Bayesian model."""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import normalize
from copy import deepcopy
import os

# class mapping
classes = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
# reverse class mapping
reverse_classes = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

# function to load the Bayesian model
def load_model():
    # 1. Prepare and read the labels histogram file

    # Assign the file to a variable
    labels_histogram = 'labels_histogram.xlsx'

    # Read the file using pandas' as a dataframe
    # sheet_name specifies the sheet to read
    # header = 0 tells pandas to consider the first line as the header
    df = pd.read_excel(labels_histogram, sheet_name="labels_histogram", header=0)

    # print the first 5 rows of the dataframe
    df.head()

    # -----------------------------------------------------------------

    # 2. Prepare data for the Bayesian Network

    # List of labels
    # first column of the dataframe
    # skip last row i.e. "total"
    labels_list = df['label'][:-1]


    # Count total Positive, Negative and Neutral emotions

    # get the value of the last row in the positive column of the dataframe
    total_positive_labels = df['positive'].iloc[-1]

    # get the value of the last row in the neutral column of the dataframe
    total_neutral_labels = df['neutral'].iloc[-1]

    # get the value of the last row in the negative column of the dataframe
    total_negative_labels = df['negative'].iloc[-1]


    # Frequencies of Positive, Negative and Neutral emotions for each label

    # get the column with the name "positive" as a numpy array
    # skip last row
    positive_ndarray = np.array(df['positive'][:-1])

    # get the column with the name "neutral" as a numpy array
    # skip last row
    neutral_ndarray = np.array(df['neutral'][:-1])

    # get the column with the name "negative" as a numpy array
    # skip last row
    negative_ndarray = np.array(df['negative'][:-1])


    # -----------------------------------------------------------------

    # 3. Model

    # Define the edges for the model

    # edge from Emotion node to each label node i.e. 809 edges
    edges_list = [("Emotion", label) for label in labels_list]
    # edge from Emotion node to CNN node
    edges_list.append(("Emotion", "CNN"))


    # Define the Model
    model = BayesianModel()


    # Add nodes and edges to the model
    # Add all the labels from labels_list as nodes
    model.add_nodes_from(labels_list)
    # Add all the edges from edges_list
    model.add_edges_from(edges_list)


    # Create the Conditional Probability Distribution Table for the Emotion node
    # Name of the node is "Emotion"
    # Total variables = 3 i.e. 1 for each emotion
    # Since, each emotion is equally likely so each will have 1/3 probability
    emotion_cpd = TabularCPD("Emotion", 3, values=[[1./3,1./3,1./3]])


    # Create the Conditional Probability Distribution Table for the CNN node

    # Calculate the conditional probability values using the confusion matrix obtained from the CNN
    # Store the confusion matrix obtained from CNN as a numpy array

    # 		    Pos		Neg		Neu
    # Pos		583.0	117.0	60.0
    # Neg		65.0	396.0	43.0
    # Neu		149.0	305.0	211.0

    cnn_confusion_matrix = np.array([[2735.0, 210.0, 430.0],
                                     [330.0, 791.0, 552.0],
                                     [374.0, 387.0, 1097.0]])
    # Normalize the confusion matrix
    cnn_confusion_matrix = normalize(cnn_confusion_matrix, axis=1, norm="l1")
    # CNN CPD values will be the transpose of the confusion matrix
    cnn_values = cnn_confusion_matrix.T


    # Name of the node is "CNN"
    # Total variables = 3 i.e. 1 for each emotion
    # Set the Emotion node as the evidence
    cnn_cpd = TabularCPD("CNN", 3, evidence=['Emotion'], evidence_card=[3], values=cnn_values)


    # Create Conditional Probability Distribution Tables for each Label node

    # create a list to store each label cpd
    label_cpd_list = []

    for i in range(len(labels_list)):
        # P(label=1|Emotion=Positive)
        p_label_1_given_emo_positive = float(positive_ndarray[i]/total_positive_labels)
        # P(label=1|Emotion=Neutral)
        p_label_1_given_emo_neutral = float(neutral_ndarray[i]/total_neutral_labels)
        # P(label=1|Emotion=Negative)
        p_label_1_given_emo_negative = float(negative_ndarray[i]/total_negative_labels)

        # if P(label=1|Emotion=Positive) is 0, set it to 0.0001 to fix the error
        if p_label_1_given_emo_positive == 0.0:
            p_label_1_given_emo_positive = 0.0001
        
        # if P(label=1|Emotion=Neutral) is 0, set it to 0.0001 to fix the error
        if p_label_1_given_emo_neutral == 0.0:
            p_label_1_given_emo_neutral = 0.0001

        # if P(label=1|Emotion=Negative) is 0, set it to 0.0001 to fix the error  
        if p_label_1_given_emo_negative == 0.0:
            p_label_1_given_emo_negative = 0.0001
        
        # P(label=0|Emotion=Positive)
        p_label_0_given_emo_positive = 1.0 - p_label_1_given_emo_positive
        # P(label=0|Emotion=Neutal)
        p_label_0_given_emo_neutral = 1.0 - p_label_1_given_emo_neutral
        # P(label=0|Emotion=Negative)
        p_label_0_given_emo_negative = 1.0 - p_label_1_given_emo_negative
        
        # Condition Probability Table for the label
        label_conditional_probability_table = [[p_label_0_given_emo_positive, p_label_0_given_emo_negative, p_label_0_given_emo_neutral], 
                                                [p_label_1_given_emo_positive, p_label_1_given_emo_negative, p_label_1_given_emo_neutral]]
        
        # generate the conditional probability table for that label
        # Name of the node is the name of the label
        # Total variables = 2 i.e. either 1 or 0
        # Set the Emotion node as the evidence
        label_cpd = TabularCPD(labels_list[i], 2, evidence=['Emotion'], evidence_card=[3], values=label_conditional_probability_table)

    #         print(cpd_label)
    #         ╒═════════╤══════════════════════╤═══════════╤═══════════════════════╕
    #         │ Emotion │ Emotion_0            │ Emotion_1 │ Emotion_2             │
    #         ├─────────┼──────────────────────┼───────────┼───────────────────────┤
    #         │ gown_0  │ 0.974076983503535    │ 1.0       │ 0.9983333333333333    │
    #         ├─────────┼──────────────────────┼───────────┼───────────────────────┤
    #         │ gown_1  │ 0.025923016496465043 │ 0.0       │ 0.0016666666666666668 │
    #         ╘═════════╧══════════════════════╧═══════════╧═══════════════════════╛

        # add it to the list
        label_cpd_list.append(label_cpd)


    # Add Conditional Probability Tables to the Model

    # Add the emotion and CNN nodes to the model
    model.add_cpds(emotion_cpd, cnn_cpd)
    # Add the cpd for each label
    for label_cpd in label_cpd_list: model.add_cpds(label_cpd)


    # Check if model is valid

    print(model.check_model())  # returns True if the model is correct

    # print the first 10 cpds from the model
    model.get_cpds()[:10]

    # return
    # 1. the model
    # 2. the labels list
    return model, labels_list


# function to perform inference on the Bayesian network
def inference(model, labels_list, labels_for_image, cnn_prediction=None):
    # Set evidences for the nodes using results from Vision API and CNN

    # if detected label is present in labels list then set that label to 1 else 0
    label_evidences = {label:(1 if label in labels_for_image else 0) for label in labels_list}

    # Initialize Variable Elimination and query

    # Set the inference method
    emotion_infer = VariableElimination(model)

    # Compute the probability of the emotions given the detected labels list
    q = emotion_infer.query(['Emotion'], evidence=label_evidences)
    # print(q['Emotion'])

    # set bayes_prediction and bayes + cnn prediction to None
    bayes_prediction, bayes_cnn_prediction = None, None
    # a dictionary to store emotion predictions predicted by the Bayesian network
    emotion_dict = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    # predictions for the emotions
    emotion_preds = q['Emotion'].values
    
    # if all the predictions are nan then set prediction label to None
    if np.isnan(emotion_preds).all():
        bayes_prediction = "None"
    else:
        # set the probability for Positive emotion for the image
        emotion_dict['Positive'] = round(emotion_preds[0], 4)
        # set the probability for Negative emotion for the image
        emotion_dict['Negative'] = round(emotion_preds[1], 4)
        # set the probability for Neutral emotion for the image
        emotion_dict['Neutral'] = round(emotion_preds[2], 4)
        # set bayes prediction to be the class with the highest probability
        bayes_prediction = classes[np.argmax(emotion_preds)]

    # a dictionary to store emotion predictions predicted by the Bayesian network + CNN node
    emotion_cnn_dict = deepcopy(emotion_dict)

    # if the CNN prediction is not None
    if cnn_prediction is not None:
        # get prediction from CNN
        label_evidences['CNN'] = reverse_classes[cnn_prediction]

        # Compute the probability of the emotions given the detected labels list
        q = emotion_infer.query(['Emotion'], evidence=label_evidences)
        # print(q['Emotion'])
        emotion_preds = q['Emotion'].values

        # if all the predictions are nan then set prediction label to None
        if np.isnan(emotion_preds).all():
            bayes_cnn_prediction = "None"
        else:
            # set the probability for Positive emotion for the image
            emotion_cnn_dict['Positive'] = round(emotion_preds[0], 4)
            # set the probability for Negative emotion for the image
            emotion_cnn_dict['Negative'] = round(emotion_preds[1], 4)
            # set the probability for Neutral emotion for the image
            emotion_cnn_dict['Neutral'] = round(emotion_preds[2], 4)
            # set bayes + CNN prediction to be the class with the highest probability
            bayes_cnn_prediction = classes[np.argmax(emotion_preds)]
    else:
        bayes_cnn_prediction = bayes_prediction

    # return -
    # i. label of the predicted emotion for the whole image (using the Bayesian Network)
    # ii. label of the predicted emotion for the whole image (using the Bayesian Network + CNN as a node)
    # iii. Bayesian predictions for the whole image
    # iv. Bayesian + CNN predictions for the whole image
    return bayes_prediction, bayes_cnn_prediction, emotion_dict, emotion_cnn_dict