#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import json
import os

from scipy.stats import uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def svm(protocol, feature_mode, df, adjust_params=False):
    # base_path
    base_path = "ml/{}/{}".format(protocol, feature_mode)
    # split the label and features
    X = df.drop('isHoneypot_new', axis=1)
    y = df['isHoneypot_new']

    # split dataset to train dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if adjust_params:
        # random search
        param_dist = {'kernel': ['linear', 'rbf'], 'C': uniform(loc=0, scale=100), 'gamma': ['scale', 'auto']}
        svc = SVC()
        random_search = RandomizedSearchCV(svc, param_distributions=param_dist, cv=5, n_iter=10, random_state=42)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)
        json_str = json.dumps(random_search.best_params_)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open("{}/params_svm.json".format(base_path), "w") as f:
            f.write(json_str)
        clf = SVC(**random_search.best_params_, random_state=0)
    else:
        # create svm classifier
        clf = SVC(kernel='linear', C=5.8083612168199465, gamma="auto", random_state=0)

    # train model
    clf.fit(X_train, y_train)

    # predict the test dataset
    y_pred = clf.predict(X_test)

    # calculate and print the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')
    result_json = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1
    }
    with open("{}/result_svm.json".format(base_path), "w") as f:
        f.write(json.dumps(result_json))