#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import pandas as pd

from ml.decision_tree import decision_tree
from ml.feature_process import feature_process
from ml.neural_networks import neural_networks
from ml.random_forest import random_forest
from ml.svm import svm

if __name__ == '__main__':
    protocols = ["atg", "s7", "modbus"]
    feature_modes = ["common_features", "all_features"]
    with_params_adjust = True

    for protocol in protocols:
        for feature_mode in feature_modes:
            # read origin data
            print("Begin to read {} data:".format(protocol))
            df = pd.read_csv('data/{}.csv'.format(protocol), sep=",")
            print(df.columns)
            print("==========================")

            # process the features
            print("Begin to process {} features:".format(protocol))
            feature_process(df, protocol)
            print("==========================")

            # read features
            print("Begin to read {} features:".format(protocol))
            if feature_mode == "all_features":
                df = pd.read_csv('ml/{}/all_features.csv'.format(protocol))
            elif feature_mode == "common_features":
                df = pd.read_csv('ml/{}/common_features.csv'.format(protocol))
            print("==========================")

            # run the algorithm
            # random forest
            print("Begin random forest:")
            random_forest(protocol, feature_mode, df, with_params_adjust)
            print("==========================")
            # decision tree
            print("Begin Decision Tree:")
            decision_tree(protocol, feature_mode, df, with_params_adjust)
            print("==========================")
            # svm
            print("Begin SVM:")
            svm(protocol, feature_mode, df, with_params_adjust)
            print("==========================")
            # neural networks
            print("Begin neural networks:")
            neural_networks(protocol, feature_mode, df, False)
            print("==========================")
