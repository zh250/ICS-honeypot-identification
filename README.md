# ICS honeypot identification


## Files and directories in the repository

### [data](./data/)
Original datasets of three ICS protocols ([ATG](https://secure.apps.nd.gov/doh/operator/Training/OperatorTraining_ATG.pdf), [modbus](https://en.wikipedia.org/wiki/Modbus), [S7](https://wiki.wireshark.org/S7comm)) after label and feature processing (Datasets were based on [yunyueye's work](https://github.com/yunyueye/honeypot), more details about datasets please refer [here](./data/datasets_preprocessing.txt), The original raw data of datasets were captured and authorized by [Ditecting](https://www.ditecting.com/) with some attributes were queried from [Shodan's API](https://developer.shodan.io/api/clients))
**for legal and ethcial reasons, all IP were hidden**

### [ml](./ml/)
the implementation of feature processing and machine learning algorithm (all were impelmented with scikit-learn)
the processed data of original datasets are also in corresponding sub-directory of this folder
The experimental results are in the folders of their corresonding algorithms of this folder, the filenames end with result, and the common features and all features are distinguished

### [results.csv](./result.csv)
The experimental results in this file just a summary of all results and compare with [Wu's research](). You can execute the program by yourself to check the result (*mention: datasets and source code may be updated. Although I will try to make sure all files in the repository are updated correctly, the result may be different. It is just a reference*)

### [main.py](./main.py)
The main entrance of all ML program (**you do not need to execute any other program in the repository, for more details about executing, please refer [here]()**), including four machine learning algorithm experiments of three protocols, distinguishes common features and all features, and supports parameter search


## Others

Originally，there is a web GUI for demonstration (like Shodan's [Honeyscore](https://honeyscore.shodan.io/)). However, without the IP address databases (they cannot be disclosed), it is meaningless to upload source code here

For researchers in the field: **[Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is unsutiable for the classifcation task here**, because it cannot make sure OS and ISP type of a captured IP mutually independent. Some cloud service providers only provide certain OSs (like Linode), and IP of cloud service providers accounts for a non-negligible proportion in these datasets

The mian innovation in this project is the feature extraction and processing method, please refer [here](./data/datasets_preprocessing.txt) for more details, and more explanations and details will be supplemented in the future


### How to execute

#### Requirements

Operating system and libraries: tested on Ubuntu 22.04 LTS with Python 3.10.12, numpy 1.20.2, pandas 2.1.3, scipy 1.11.4, sklearn 1.4.1.post1 and mocOS 12.7.4 (Intel) with Python 3.9.6, numpy 1.24.3, pandas 2.0.1, scipy 1.10.1, sklearn 1.2.2

#### Executing steps

1. ``
2. ``
3. ``


## Future works
The project can be regarded as a work of cyber assets discovery and identification as well since it is a multi-class classifcation

More ICS and IoT and even some IT protocols which are used by CPSs may be studied in the future.

Although it demonstrated a relatively good preformance, but preformance in modbus is obviously worse than ATG and S7. Results of permutation feature importance have not been explained with some other phenomena have not been explained till now.

These phenomena indicated futher study and improvement of feature extraction and processing method are necessary. Some ideas were already in my mind, but immature.
