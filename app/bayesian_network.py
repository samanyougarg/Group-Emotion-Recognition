#!/usr/bin/env python
# coding: utf-8

# # Bayesian Network

# ## 1. Setup

# ### 1.1 Import Required Libraries
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import normalize
from copy import deepcopy
import os

classes = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
reverse_classes = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

def load_model():
    # ### 1.2 Prepare and read the labels histogram file
    # Assign the file to a variable
    labels_histogram = os.getcwd() + '/app/labels_histogram.xlsx'

    # Read the file using pandas' as a dataframe
    # sheet_name specifies the sheet to read
    # header = 0 tells pandas to consider the first line as the header
    df = pd.read_excel(labels_histogram, sheet_name="labels_histogram", header=0)

    # print the first 5 rows of the dataframe
    df.head()


    # ## 2. Prepare data for the Bayesian Network

    # ### 2.1 List of labels
    # first column of the dataframe
    # skip last row i.e. "total"
    labels_list = df['label'][:-1]


    # ### 2.2 Count total Positive, Negative and Neutral emotions
    # get the value of the last row in the positive column of the dataframe
    total_positive_labels = df['positive'].iloc[-1]

    # get the value of the last row in the neutral column of the dataframe
    total_neutral_labels = df['neutral'].iloc[-1]

    # get the value of the last row in the negative column of the dataframe
    total_negative_labels = df['negative'].iloc[-1]


    # ### 2.3 Frequencies of Positive, Negative and Neutral emotions for each label
    # get the column with the name "positive" as a numpy array
    # skip last row
    positive_ndarray = np.array(df['positive'][:-1])

    # get the column with the name "neutral" as a numpy array
    # skip last row
    neutral_ndarray = np.array(df['neutral'][:-1])

    # get the column with the name "negative" as a numpy array
    # skip last row
    negative_ndarray = np.array(df['negative'][:-1])


    # ## 3. Model

    # ### 3.1 Define the edges for the model
    # edge from Emotion node to each label node i.e. 809 edges
    edges_list = [("Emotion", label) for label in labels_list]
    # edge from Emotion node to CNN node
    edges_list.append(("Emotion", "CNN"))


    # ### 3.2 Define the Model
    model = BayesianModel()


    # ### 3.3 Add nodes and edges to the model
    # Add all the labels from labels_list as nodes
    model.add_nodes_from(labels_list)
    # Add all the edges from edges_list
    model.add_edges_from(edges_list)


    # ### 3.4 Create the Conditional Probability Distribution Table for the Emotion node
    # Name of the node is "Emotion"
    # Total variables = 3 i.e. 1 for each emotion
    # Since, each emotion is equally likely so each will have 1/3 probability
    emotion_cpd = TabularCPD("Emotion", 3, values=[[1./3,1./3,1./3]])


    # ### 3.5 Create the Conditional Probability Distribution Table for the CNN node

    # #### 3.5.1 Calculate the conditional probability values using the confusion matrix obtained from the CNN
    # Store the confusion matrix obtained from CNN as a numpy array
    cnn_confusion_matrix = np.array([[470.0,92.0,198.0],
                                    [38.0, 336.0, 130.0],
                                    [46.0, 201.0, 418.0]])
    # Normalize the confusion matrix
    cnn_confusion_matrix = normalize(cnn_confusion_matrix, axis=1, norm="l1")
    # CNN CPD values will be the transpose of the confusion matrix
    cnn_values = cnn_confusion_matrix.T


    # Name of the node is "CNN"
    # Total variables = 3 i.e. 1 for each emotion
    # Set the Emotion node as the evidence
    cnn_cpd = TabularCPD("CNN", 3, evidence=['Emotion'], evidence_card=[3], values=cnn_values)


    # ### 3.6 Create Conditional Probability Distribution Tables for each Label node

    # create a list to store each label cpd
    label_cpd_list = []

    for i in range(len(labels_list)):
        # P(label=1|Emotion=Positive)
        p_label_1_given_emo_positive = float(positive_ndarray[i]/total_positive_labels)
        # P(label=1|Emotion=Neutral)
        p_label_1_given_emo_neutral = float(neutral_ndarray[i]/total_neutral_labels)
        # P(label=1|Emotion=Negative)
        p_label_1_given_emo_negative = float(negative_ndarray[i]/total_negative_labels)
        
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


    # ### 3.7 Add Conditional Probability Tables to the Model

    # Add the emotion and CNN nodes to the model
    model.add_cpds(emotion_cpd, cnn_cpd)
    # Add the cpd for each label
    for label_cpd in label_cpd_list: model.add_cpds(label_cpd)


    # ### 3.8 Check if model is valid

    print(model.check_model())  # returns True if the model is correct

    # print the first 10 cpds from the model
    model.get_cpds()[:10]

    return model, labels_list


def inference(model, labels_list, labels_for_image, cnn_prediction=None):
    # ## 4. Inference

    # ### 4.1 Set evidences for the nodes using results from Vision API and CNN

    # if detected label is present in labels list then set that label to 1 else 0
    label_evidences = {label:(1 if label in labels_for_image else 0) for label in labels_list}

    # #### 4.2 Initialize Variable Elimination and query
    # Set the inference method
    emotion_infer = VariableElimination(model)

    # Compute the probability of the emotions given the detected labels list
    q = emotion_infer.query(['Emotion'], evidence=label_evidences)
    print(q['Emotion'])

    emotion_dict = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    emotion_cnn_dict = deepcopy(emotion_dict)
    emotion_preds = q['Emotion'].values
    emotion_dict['Positive'] = round(emotion_preds[0], 4)
    emotion_dict['Negative'] = round(emotion_preds[1], 4)
    emotion_dict['Neutral'] = round(emotion_preds[2], 4)

    if cnn_prediction is not None:
        # get prediction from CNN
        label_evidences['CNN'] = reverse_classes[cnn_prediction]

        # Compute the probability of the emotions given the detected labels list
        q = emotion_infer.query(['Emotion'], evidence=label_evidences)
        print(q['Emotion'])

        emotion_preds = q['Emotion'].values
        emotion_cnn_dict['Positive'] = round(emotion_preds[0], 4)
        emotion_cnn_dict['Negative'] = round(emotion_preds[1], 4)
        emotion_cnn_dict['Neutral'] = round(emotion_preds[2], 4)

        return classes[np.argmax(emotion_preds)], emotion_dict, emotion_cnn_dict

    return classes[np.argmax(emotion_preds)], emotion_dict, None