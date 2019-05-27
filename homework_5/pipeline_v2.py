import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import * 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

################################################################################
##### PROCESSING FUNCTIONS 
################################################################################

# Adopted from https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding

def pre_process(train_df, test_df, target_col, feature_cols): 
    X_train, y_train = generate_features(train_df, target_col, feature_cols)
    X_test, y_test = generate_features(test_df, target_col, feature_cols)

    # Excludes values in test data not seen in training data 
    X_train, X_test = X_train.align(X_test, join='left', axis=1) 
    X_test.fillna(0, inplace=True)
    return X_train, X_test, y_train, y_test 

def generate_features(df, target_col, feature_cols): 
    X_df = df[feature_cols]
    y_df = df[target_col]
    X_df = X_df.fillna(X_df.mean())
    X_df = pd.get_dummies(X_df, dummy_na=True)
    return X_df, y_df 
    
################################################################################
##### VALIDATION FUNCTIONS 
################################################################################

def temporal_split(df, start_date, end_date, test_window, prediction_horizon, date_col, i): 
    df[date_col] = pd.to_datetime(df[date_col])
    train_start = pd.to_datetime(start_date)
    test_start = pd.to_datetime(start_date) + (i * test_window)
    train_end = test_start - prediction_horizon
    test_end = test_start + test_window
    
    df['train'] = np.where((df[date_col] > train_start) & (df[date_col] < train_end), 1, 0)
    df['test'] = np.where((df[date_col] > test_start) & (df[date_col] < test_end), 1, 0)

    train_df = df.loc[df['train']==1]
    test_df = df.loc[df['test']==1]
    return train_df, test_df

################################################################################
##### CLASSIFIER FUNCTIONS 
################################################################################

def build_classifier(clf, model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    if clf == 'SVM': 
        scores = model.decision_function(X_test)
    else: 
        scores = model.predict_proba(X_test)[:,1]
    return scores 

################################################################################
##### EVALUATION FUNCTIONS 
################################################################################

# Adopted from https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py

def joint_sort_descending(l1, l2):
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def metrics_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))    
    preds_at_k = generate_binary_at_k(y_scores, k)    

    accuracy = accuracy_score(y_true, preds_at_k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    auc_roc = roc_auc_score(y_true, y_scores)
    return [k, accuracy, precision, recall, f1, auc_roc]

def plot_precision_recall_n(y_true, y_prob):
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_prob)
    for value in pr_thresholds:
        num_above_thresh = len(y_prob[y_prob>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    plt.show()
