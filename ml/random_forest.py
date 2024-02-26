#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def random_forest(protocol, feature_mode, df, adjust_params=False):
    # base_path
    base_path = "ml/{}/{}".format(protocol, feature_mode)
    # split the label and features
    X = df.drop('isHoneypot_new', axis=1)
    y = df['isHoneypot_new']
    print(X.shape)

    # print the distribution of label
    print(f'Class 0: {sum(y == 0)} samples')
    print(f'Class 1: {sum(y == 1)} samples')
    print(f'Class 2: {sum(y == 2)} samples')
    print(f'Class 3: {sum(y == 3)} samples')

    # split dataset to train dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if adjust_params:
        # adjust the params through grid
        rfc = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [5, 10, 20, 25, 30]}
        grid_search = GridSearchCV(rfc, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        json_str = json.dumps(grid_search.best_params_)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open("{}/params_random_forest.json".format(base_path), "w") as f:
            f.write(json_str)
        clf = RandomForestClassifier(**grid_search.best_params_, random_state=0)
    else:
        # create rf classifier, n_estimators means the number of dt, max_depth means the depth of tree, random_state is the seed of random
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

    # train model
    clf.fit(X_train, y_train)

    # calculte the importance of features
    importances = clf.feature_importances_
    # print the score of each feature
    importances_json = {}
    for i, importance in enumerate(importances):
        key = f"Feature {i}"
        importances_json[key] = importance
        print(f'Feature {i}: {importance}')
    with open("{}/feature_importances.json".format(base_path), "w") as f:
        f.write(json.dumps(importances_json))

    # predict the test dataset
    y_pred = clf.predict(X_test)

    # calculate and print the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
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
    with open("{}/result_random_forest.json".format(base_path), "w") as f:
        f.write(json.dumps(result_json))
