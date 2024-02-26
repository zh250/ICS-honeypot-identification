#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import os

import pandas as pd
import numpy as np

base_path = ""

def feature_process(df, protocol):
    # init params
    global base_path
    base_path = "ml/{}".format(protocol)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    print(protocol)
    if protocol == "atg":
        atg_feature_process(df)
    elif protocol == "s7":
        s7_feature_process(df)
    elif protocol == "modbus":
        modbus_feature_process(df)

# process dataset of ATG
def atg_feature_process(df):

    # honeypot label
    # label, 'unknown'-0 'honeypot'-1 'ICS'-2 'other'-3 'ambiguous'-4
    label_mapping = {'unknown': 0, 'honeypot': 1, 'ICS': 2, 'other': 3, 'ambiguous': 4}
    df['isHoneypot_new'] = df['isHoneypot_new'].map(label_mapping)
    print("isHoneypot_new: " + str(df["isHoneypot_new"].unique()))
    
    # gaspot features

    # feature 0-3, when it is SUPER, UNLEAD, DIESEL, PREMIUM separately
    # each of them mark as 1, otherwise, mark as 0
    print("atgSUPER: " + str(df["atgSUPER"].unique()))
    print("atgUNLEAD: " + str(df["atgUNLEAD"].unique()))
    print("atgDIESEL: " + str(df["atgDIESEL"].unique()))
    print("atgPREMIUM: " + str(df["atgPREMIUM"].unique()))

    # feature 4, if exist any product | ULLAGE1 - ULLAGE2 | != | VOLUMETC1 - VOLUMETC2 |
    # mark as Flase
    # just use the result from atgTwoTimesCompare column
    df['atgTwoTimesCompare'] = df['atgTwoTimesCompare'].astype(int)
    print("atgTwoTimesCompare: " + str(df["atgTwoTimesCompare"].unique()))
    
    # common features

    # feature 5: hop number 13<hopNum<30, split every 5 hops as a group
    # 11-15 as 0, 16-20 as 1, 21-25 as 2, 26-30 as 3
    bins = [11, 16, 21, 26, 30]
    labels = [0, 1, 2, 3]
    df['hopNum'] = pd.cut(df['hopNum'], bins=bins, labels=labels).astype(int)
    print("hopNum: " + str(df["hopNum"].unique()))

    # feature 6: open port number: use origin value
    print("OpenPortNum: " + str(df["OpenPortNum"].unique()))

    # feature 7: OS feature
    # Linux (exclude embedded and android) mark as 0
    # Windows (include server but exclude embedded) mark as 1
    # Unix mark as 2,
    # common proprietary OS (include Windows embedded, Linux embedded and Android) mark as 3
    # ICS proprietary OS (label as ICS in the chart) mark as 4
    # some devices may contain different OS feature at the same time
    # in such case, it will be marked as (ambiguous in the chart) 5
    # unknown mark as 6. Default is unknow
    os_mapping = {'Linux': 0, 'Windows': 1, 'Unix': 2, 'proprietary': 3, 'ICS': 4, 'ambiguous': 5, 'unknown': 6}
    df['OS_conclusion'] = df['OS_conclusion'].map(os_mapping).fillna(6).astype(int)
    print("OS_conclusion: " + str(df["OS_conclusion"].unique()))

    # feature 8: ISP feature
    # mark cloud and VPS, hosting and datacenter as 0
    # University (mainly refer to University of Maryland (label as UoM in the chart)) as 1
    # other ordinary ISP (telecom in the chart) mark as 2
    # education and research network (except UoM, edu in the dataset) mark as 3
    # industrial company (industry in the chart) mark as 4
    # unknown mark as 5, default is unknown
    # ambiguous mark as 6
    isp_mapping = {'cloud': 0, 'VPS': 0, 'hosting': 0, 'datacenter': 0, 'UoM': 1, "telecom": 2, "edu": 3,
                    'industry': 4, 'unknown': 5, 'ambiguous': 6, np.nan: 5}
    df['ISP_conclusion'] = df['ISP_conclusion'].map(isp_mapping).fillna(5).astype(int)
    print("ISP_conclusion: " + str(df["ISP_conclusion"].unique()))

    # final df with all features for training
    all_features = df.loc[:,
                   ['isHoneypot_new', 'atgSUPER', 'atgUNLEAD', 'atgDIESEL', 'atgPREMIUM', 'atgTwoTimesCompare',
                    'hopNum', 'OpenPortNum', 'OS_conclusion', 'ISP_conclusion']]
    print(all_features)
    all_features.to_csv('{}/all_features.csv'.format(base_path), index=False)

    # final df with common features for training
    common_features = df.loc[:, ['isHoneypot_new', 'hopNum', 'OpenPortNum', 'OS_conclusion', 'ISP_conclusion']]
    print(common_features)
    common_features.to_csv('{}/common_features.csv'.format(base_path), index=False)


# process dataset of S7
def s7_feature_process(df):
    
    # honeypot label
    # label, 'unknown'-0 'honeypot'-1 'ICS'-2 'other'-3 'ambiguous'-4
    print("isHoneypot_new: " + str(df["isHoneypot_new"].unique()))
    label_mapping = {'unknown': 0, 'honeypot': 1, 'ICS': 2, 'other': 3, 'ambiguous': 4, np.nan: 0}
    df['isHoneypot_new'] = df['isHoneypot_new'].map(label_mapping)
    print("isHoneypot_new: " + str(df["isHoneypot_new"].unique()))
    
    # S7 conpot features

    # feature 0, for PlantIdentification, mark empty, "Mouser Factory", "DoE Water Service"
    # as 0,1,2 separately, others as 3
    mapping = {"": 0, np.nan: 0, "Mouser Factory": 1, "DoE Water Service": 2}
    df["s7PlantIdentification"] = df["s7PlantIdentification"].map(mapping).fillna(3).astype(int)
    print("s7PlantIdentification: " + str(df["s7PlantIdentification"].unique()))

    # feature 1, for NameOfThePLC, mark empty，Technodrome，SAAP7-SERVER, SIMATIC 300(1), PC35xV
    # as 0,1,2,3,4 separately, others as 5
    mapping = {"": 0, np.nan: 0, "Technodrome": 1, "SAAP7-SERVER": 2, "SIMATIC 300(1)": 3, "PC35xV": 4}
    df["s7NameOfThePLC"] = df["s7NameOfThePLC"].map(mapping).fillna(5).astype(int)
    print("s7NameOfThePLC: " + str(df["s7NameOfThePLC"].unique()))

    # feature 2, for SerialNumberOfModule, mark empty, 88111222, "S C-C2UR28922012"
    # as 0,1,2 separately, others as 3
    mapping = {"": 0, np.nan: 0, "88111222": 1, "S C-C2UR28922012": 2}
    df["s7SerialNumberOfModule"] = df["s7SerialNumberOfModule"].map(mapping).fillna(3).astype(int)
    print("s7SerialNumberOfModule: " + str(df["s7SerialNumberOfModule"].unique()))

    # feature 3, for Time5Later (namely s7time5After in chart)
    # indicates whether the host can response after 5s, boolean
    # True (in the chart) represent SYN (keep data communication)
    # Flase (in the chart) represent FIN+RST (data link rest by honeypot)
    df['s7time5After'] = df['s7time5After'].fillna(0).astype(int)
    print("s7time5After: " + str(df["s7time5After"].unique()))

    # feature 4: s7ResponseTime
    # the responsetime is encode to 7 groups from 0s and split every 0.2s as a group
    # some cells in this column is empty, it may indicate no response, and mark empty as 0
    df["s7ResponseTime"] = df["s7ResponseTime"].fillna(-1).astype(float)
    bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 10]
    labels = [0, 1, 2, 3, 4, 5, 6]
    df["s7ResponseTime"] = pd.cut(df["s7ResponseTime"], bins=bins, labels=labels, include_lowest=True).astype(int)
    print("s7ResponseTime: " + str(df["s7ResponseTime"].unique()))

    # common features

    # feature 5: hop number, split the number into 4 groups
    # 13<hopNum<30, split every 5 hops as a group
    # mark 11-15 as 0, 16-20 as 1, 21-25 as 2, 26-30 as 3
    bins = [11, 16, 21, 26, 30]
    labels = [0, 1, 2, 3]
    df["hopNum"] = pd.cut(df["hopNum"], bins=bins, labels=labels, include_lowest=True).fillna(3).astype(int)
    print("hopNum: " + str(df["hopNum"].unique()))

    # feature 6: open port number: origin value
    df["OpenPortNum"] = df["OpenPortNum"].fillna(6).astype(int)
    print("OpenPortNum: " + str(df["OpenPortNum"].unique()))

    # feature 7: OS feature
    # Linux (exclude embedded and android) mark as 0
    # Windows (include server but exclude embedded) mark as 1
    # Unix mark as 2,
    # common proprietary OS (include Windows embedded, Linux embedded and Android) mark as 3
    # ICS proprietary OS (label as ICS in the chart) mark as 4
    # some devices may contain different OS characteristics at the same time,
    # in such case, it will be marked as (ambiguous in the chart) 5
    # unknown mark as 6. Default is unknow
    os_mapping = {'Linux': 0, 'Windows': 1, 'Unix': 2, 'proprietary': 3, 'ICS': 4, 'ambiguous': 5, 'unknown': 6}
    df['OS_conclusion'] = df['OS_conclusion'].map(os_mapping).fillna(6).astype(int)
    print("OS_conclusion: " + str(df["OS_conclusion"].unique()))

    # feature 8: ISP feature
    # mark cloud and VPS, hosting and datacenter as 0
    # mark University (mainly refer to University of Maryland (label as UoM in the chart)) as 1
    # other ordinary ISP (telecom in the chart) mark as 2
    # education and research network (except UoM, edu in the dataset) mark as 3,
    # industrial company (industry in the chart) mark as 4
    # unknown mark as 5
    # ambiguous mark as 6. Default is unknown
    isp_mapping = {'cloud': 0, 'VPS': 0, 'hosting': 0, 'datacenter': 0, 'UoM': 1, "telecom": 2, "edu": 3, 
                    'industry': 4, 'unknown': 5, 'ambiguous': 6, np.nan: 5}
    df['ISP_conclusion'] = df['ISP_conclusion'].map(isp_mapping).fillna(5).astype(int)
    print("ISP_conclusion: " + str(df["ISP_conclusion"].unique()))

    # final df with all features for training
    all_features = df.loc[:,
                   ['isHoneypot_new', 's7PlantIdentification', 's7NameOfThePLC', 's7SerialNumberOfModule', 's7time5After', 's7ResponseTime',
                    'hopNum', 'OpenPortNum', 'OS_conclusion', 'ISP_conclusion']]
    print(all_features)
    all_features.to_csv('{}/all_features.csv'.format(base_path), index=False)

    # final df with common features for training
    common_features = df.loc[:, ['isHoneypot_new', 'hopNum', 'OpenPortNum', 'OS_conclusion', 'ISP_conclusion']]
    print(common_features)
    common_features.to_csv('{}/common_features.csv'.format(base_path), index=False)


# process dataset of modbus
def modbus_feature_process(df):

    # honeypot label
    # label, 'unknown'-0 'honeypot'-1 'ICS'-2 'other'-3 'ambiguous'-4
    label_mapping = {'unknown': 0, 'honeypot': 1, 'ICS': 2, 'other': 3, 'ambiguous': 4}
    df['IsHoneypot_new'] = df['IsHoneypot_new'].map(label_mapping).fillna(0).astype(int)
    df = df.rename(columns={"IsHoneypot_new": "isHoneypot_new"})
    print("isHoneypot_new: " + str(df["isHoneypot_new"].unique()))

    # modbus conpot features

    # feature 0, for modbusReadRegister
    # if it is "connection failed", mark as 0
    # if it is "25|2|88", mark as 1,
    # if it is "0|0|0", mark as 2
    # if it contain other 3 numbers, mark as 3
    mapping = {"connection failed": 0, "25|2|88": 1, "0|0|0": 2}
    df["modbusReadRegister"] = df["modbusReadRegister"].map(mapping).fillna(3).astype(int)
    print("modbusReadRegister: " + str(df["modbusReadRegister"].unique()))

    # feature 1: response time of receiving error code
    # devide the time by k-bins algorithm
    df["modbusErrorRequestTime (ms)"] = df["modbusErrorRequestTime (ms)"].fillna(-1).astype(float)
    bins = [-1]
    for i in range(0, 20):
        bins.append(i*120)
    labels = [i for i in range(0, 20)]
    df["modbusErrorRequestTime (ms)"] = pd.cut(df["modbusErrorRequestTime (ms)"], bins=bins, labels=labels, include_lowest=True).astype(int)
    print("modbusErrorRequestTime (ms): " + str(df["modbusErrorRequestTime (ms)"].unique()))

    # common features
    
    # feature 2: hopNum
    # split hop number into 4 groups
    # 13<hopNum<30, split every 5 hops as a group
    # mark 11-15 as 0, 16-20 as 1, 21-25 as 2, 26-30 as 3
    bins = [11, 16, 21, 26, 30]
    labels = [0, 1, 2, 3]
    df['hopNum'] = pd.cut(df['hopNum'], bins=bins, labels=labels).astype(int)
    print("hopNum: " + str(df["hopNum"].unique()))

    # feature 3: open port number: origin value
    print("OpenPortNum: " + str(df["OpenPortNum"].unique()))

    # feature 4: OS
    # Linux (exclude embedded and android) mark as 0
    # Windows (include server but exclude embedded) mark as 1
    # Unix mark as 2,
    # common proprietary OS (include Windows embedded, Linux embedded and Android) mark as 3
    # ICS proprietary OS (label as ICS in the chart) mark as 4
    # some devices may contain different OS features at the same time
    # in such case, it will be marked as (ambiguous in the chart) 5
    # unknown mark as 6. Default is unknown
    os_mapping = {'Linux': 0, 'Windows': 1, 'Unix': 2, 'proprietary': 3, 'ICS': 4, 'ambiguous': 5, 'unknown': 6}
    df['OS_conclusion'] = df['OS_conclusion'].map(os_mapping).fillna(6).astype(int)
    print("OS_conclusion: " + str(df["OS_conclusion"].unique()))

    # feature 5: ISP
    # mark cloud and VPS, hosting and datacenter as 0
    # University (mainly refer to University of Maryland (label as UoM in the chart)) as 1
    # other ordinary ISP (telecom in the chart) mark as 2
    # education and research network (except UoM, edu in the dataset) mark as 3,
    # industrial company (industry in the chart) mark as 4
    # unknown mark as 5. Default is unknown
    # ambiguous mark as 6
    isp_mapping = {'cloud': 0, 'VPS': 0, 'hosting': 0, 'datacenter': 0, 'UoM': 1, "telecom": 2, "edu": 3,
                   'industry': 4, 'unknown': 5, 'ambiguous': 6, np.nan: 5}
    df['ISP_conclusion'] = df['ISP_conclusion'].map(isp_mapping).fillna(5).astype(int)
    print("ISP_conclusion: " + str(df["ISP_conclusion"].unique()))

    # final df with all features for training
    all_features = df.loc[:,
                   ['isHoneypot_new', 'modbusReadRegister', 'modbusErrorRequestTime (ms)',
                    'hopNum', 'OpenPortNum', 'OS_conclusion', 'ISP_conclusion']]
    print(all_features)
    all_features.to_csv('{}/all_features.csv'.format(base_path), index=False)

    # final df with common features for training
    common_features = df.loc[:, ['isHoneypot_new', 'hopNum', 'OpenPortNum', 'OS_conclusion', 'ISP_conclusion']]
    print(common_features)
    common_features.to_csv('{}/common_features.csv'.format(base_path), index=False)