#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

def decision_tree(protocol, feature_mode, df, adjust_params=False):
    # base_path
    base_path = "ml/{}/{}".format(protocol, feature_mode)
    # split the label and features
    X = df.drop('isHoneypot_new', axis=1)
    y = df['isHoneypot_new']

    # split dataset to train dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if adjust_params:
        # adjust the params through random
        param_dist = {'max_depth': randint(1, 30), 'min_samples_split': randint(2, 20), 'min_samples_leaf': randint(1, 10)}
        dtc = DecisionTreeClassifier()
        random_search = RandomizedSearchCV(dtc, param_distributions=param_dist, cv=5, n_iter=10, random_state=42)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)
        json_str = json.dumps(random_search.best_params_)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open("{}/params_decision_tree.json".format(base_path), "w") as f:
            f.write(json_str)
        clf = DecisionTreeClassifier(**random_search.best_params_, random_state=0)
    else:
        # create dt classifier
        clf = DecisionTreeClassifier(max_depth=22, min_samples_leaf=5, min_samples_split=3, random_state=0)

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
    with open("{}/result_decision_tree.json".format(base_path), "w") as f:
        f.write(json.dumps(result_json))
