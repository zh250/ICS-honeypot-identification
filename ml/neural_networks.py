#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import json
import os

from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def neural_networks(protocol, feature_mode, df, adjust_params=False):
    # base_path
    base_path = "ml/{}/{}".format(protocol, feature_mode)
    # split the label and features
    X = df.drop('isHoneypot_new', axis=1)
    y = df['isHoneypot_new']
    print(X.shape)

    # split dataset to train dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if adjust_params:
        # adjust the params through random
        param_dist = {'hidden_layer_sizes': [(randint.rvs(16, 256),) * n for n in range(1, 6)],
                      'activation': ['relu', 'logistic', 'tanh'],
                      'solver': ['adam', 'sgd'],
                      'alpha': [10 ** -x for x in range(1, 6)],
                      'learning_rate': ['constant', 'adaptive']}

        mlp = MLPClassifier(max_iter=10000)
        random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, cv=5, n_iter=10, random_state=42)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)
        json_str = json.dumps(random_search.best_params_)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open("{}/params_neural_networks.json".format(base_path), "w") as f:
            f.write(json_str)
        clf = MLPClassifier(**random_search.best_params_, random_state=0)
    else:
        # create nn classifier
        # {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (22, 22), 'alpha': 0.0001, 'activation': 'tanh'}
        clf = MLPClassifier(hidden_layer_sizes=(22, 22), solver="adam", learning_rate="adaptive", alpha=0.01,
                            activation="tanh", max_iter=1000, random_state=0)

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
    with open("{}/result_neural_networks.json".format(base_path), "w") as f:
        f.write(json.dumps(result_json))
