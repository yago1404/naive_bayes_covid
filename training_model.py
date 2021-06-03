import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle

def training():
    def filter_label_tag(label_list):
        if label_list[0] == 1:
            return 'Pouca'
        if label_list[1] == 1:
            return 'Moderada'
        if label_list[2] == 1:
            return 'Nemhuma'
        if label_list[3] == 1:
            return 'Alta'

    dataframe = pd.read_csv('Cleaned-Data.csv')
    severity_columns = dataframe.filter(like='Severity_').columns

    dataframe['Condition'] = dataframe[severity_columns].values.tolist()

    dataframe_list = dataframe.values.tolist()

    training_features_list = []
    training_labels_list = []
    correction_features_list = []
    correction_labels_list = []
    index = 0

    classifier = GaussianNB()

    for data in dataframe_list:
        features_frame = data[0:8]
        label_frame = filter_label_tag(data[27])
        if index < 252800:
            training_features_list.append(features_frame)
            training_labels_list.append(label_frame)
        else:
            correction_features_list.append(features_frame)
            correction_labels_list.append(label_frame)
        index += 1

    classifier.fit(training_features_list, training_labels_list)

    f = open('my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()