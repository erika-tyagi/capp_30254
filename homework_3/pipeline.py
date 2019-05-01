'''
Assignment 3: Improving the Pipeline
Part 1: Coding 

The goal of this assignment is to build a modular machine learning pipeline. 

This pipeline includes the following classifiers: 
	- Logistic Regression
	- K-Nearest Neighbor
	- Decision Trees
	- SVM
	- Random Forests
	- Gradient Boosting 

And the following evaluation metrics: 
	- Accuracy
	- Precision (at different levels)
	- Recall (at different levels)
	- F1
	- Area Under Curve
	- Precision-Recall Curves 
    - ROC Curves
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_X_y

from sklearn.model_selection import train_test_split 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

################################################################################
##### VALIDATION FUNCTIONS 
################################################################################

def split(df, feature_cols, target_col, test_size, seed): 
    '''
    Splits data into X_train, X_test, y_train, and y_test data frames. 
    Parameters (and examples) include: 
        - test_size (proportion of data in test): 0.3
        - seed (random seed): 123 
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_col], test_size=test_size, random_state=seed)    
    return X_train, X_test, y_train, y_test

def temporal_split(df, feature_cols, target_col, date_col, test_start, test_end): 
    '''
    Splits data into X_train, X_test, y_train, and y_test data frames using 
    temporal validation where the testing data is between test_start and test_end
    in date_col. Assumes training data is all data before testing data. 
    '''
    df['train'] = np.where(df[date_col] < test_start, 1, 0)
    df['test'] = np.where((df[date_col] < test_end) & (df[date_col] >= test_start), 1, 0)    
    X_train = df.loc[df['train']==1, feature_cols]
    X_test = df.loc[df['test']==1, feature_cols]
    y_train = df.loc[df['train']==1, target_col]
    y_test = df.loc[df['test']==1, target_col]
    return X_train, X_test, y_train, y_test


################################################################################
##### CLASSIFIER FUNCTIONS #####
################################################################################

def build_Baseline(X_train, y_train, X_test, y_test, strategy): 
    '''
    Trains, fits, and evaluates a dummy/baseline classifer using X_train, y_train, X_test, y_test. 
    Parameters (and examples) include: 
        - strategy (to generate predictions): stratified, most_frequent, prior, uniform
    '''
    X_train, y_train = check_X_y(X_train, y_train)
    model = DummyClassifier(strategy=strategy)
    model.fit(X_train, y_train)
    return evaluate_baseline(model, X_test, y_test)

def build_LogReg(X_train, y_train, X_test, solver, penalty, C, threshold): 
    '''
    Trains and fits a logistic regression model using X_train, y_train, and X_test.  
    Parameters (and examples) include: 
        - solver (optimization algorithm): liblinear
        - penalty (penalization norm): l1, l2
        - C (regularization strength): 10**-2, 10**-1, 1 , 10, 10**2
        - threshold (to convert scores to labels): 0.5 
    '''
    model = LogisticRegression(solver=solver, penalty=penalty, C=C)
    model.fit(X_train, y_train)
    return predict(model, X_test, threshold)

def build_KNN(
    X_train, y_train, X_test, n_neighbors, distance_metric, weight, threshold): 
    '''
    Trains and fits a k-nearest neighbors model using X_train, y_train, and X_test.
    Parameters (and examples) include: 
        - n_neighbors (number of neighbors: 5  
        - distance_metric (distance metric): euclidean, manhattan, chebyshev, minkowski 
        - weight (weight function): uniform, distance 
        - threshold (to convert scores to labels): 0.5 
    '''
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric=distance_metric, weights=weight)
    model.fit(X_train, y_train)
    return predict(model, X_test, threshold)

def build_DecTree(
    X_train, y_train, X_test, criterion, max_depth, min_samples_leaf, threshold): 
    '''
    Trains and fits a decision tree classifier using X_train, y_train, and X_test.
    Parameters (and examples) include: 
        - criterion (split quailty criterion): gini, entropy 
        - max_depth (max tree depth): 5 
        - min_samples_leaf (min samples required for a leaf): 10 
        - threshold (to convert scores to labels): 0.5 
    '''
    model = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return predict(model, X_test, threshold)

def build_SVM(X_train, y_train, X_test, tol, loss, C, threshold):
    '''
    Trains and fits a linear support vector classifier using X_train, y_train, and X_test.
    Parameters (and examples) include: 
        - tol (stopping criteria tolerance): 1e-5, 1e-4
        - loss (loss function): hinge, squared_hinge
        - C (penalty parameter): 10**-2, 10**-1
        - threshold (to convert scores to labels): 0.5 
    '''
    model = LinearSVC(tol=tol, loss=loss, C=C)
    model.fit(X_train, y_train)
    confidence_score = model.decision_function(X_test)
    pred_labels = [1 if x > threshold else 0 for x in confidence_score] 
    return confidence_score, pred_labels

def build_RandForest(
    X_train, y_train, X_test, n_estimators, criterion, max_depth, min_samples_leaf, threshold): 
    '''
    Trains and fits a random forest classifier using X_train, y_train, and X_test.
    Parameters (and examples) include: 
        - n_estimators (number of trees in the forest): 10 
        - criterion (split quailty criterion): gini, entropy 
        - max_depth (max tree depth): 5 
        - min_samples_leaf (min samples required for a leaf): 10 
        - threshold (to convert scores to labels): 0.5 
    '''
    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return predict(model, X_test, threshold)

def build_GradBoost(X_train, y_train, X_test, n_estimators, learning_rate, threshold): 
    '''
    Trains and fits a gradient boosting classifier using X_train, y_train, and X_test.
    Parameters (and examples) include: 
        - n_estimators (number of boosting stages): 100 
        - learning_rate (contribution of each tree): 0.1 
        - threshold (to convert scores to labels): 0.5 
    '''
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return predict(model, X_test, threshold)


################################################################################
##### EVALUATION FUNCTIONS #####
################################################################################

def predict(model, X_test, threshold): 
    '''
    Returns prediction scores and labels (at the threshold) testing the model on X_test. 
    '''
    pred_scores = model.predict_proba(X_test)
    pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
    return pred_scores, pred_labels

def evaluate(y_test, pred_labels, pred_scores, include_ROC): 
    '''
    Returns a dictionary containing a variety of evaluation metrics. 
    '''
    evaluation_dict = {
        'accuracy': accuracy_score(y_test, pred_labels), 
        'precision': precision_score(y_test, pred_labels), 
        'recall': recall_score(y_test, pred_labels), 
        'f1': f1_score(y_test, pred_labels), 
        'confusion_matrix': confusion_matrix(y_test, pred_labels).ravel()}
    if include_ROC: 
        evaluation_dict['roc_auc'] = roc_auc_score(y_test, pred_scores[:,1])
    return evaluation_dict

def precision_recall(y_test, pred_scores):
    '''
    Returns a precision-recall curve. 
    '''
    print('Precision-Recall Curve:')
    precision, recall, _ = precision_recall_curve(y_test, pred_scores[:,1])
    plt.plot(recall, precision, marker='.')
    plt.show()

def roc(y_test, pred_scores): 
    '''
    Returns a ROC curve. 
    '''
    print('ROC Curve:')
    fpr, tpr, _ = roc_curve(y_test, pred_scores[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show()

def evaluate_baseline(model, X_test, y_test): 
    '''
    Returns a dictionary containing a variety of evaluation metrics 
    for a dummy/baseline classifer. 
    '''
    pred_labels = model.predict(X_test)
    evaluation_dict = {
        'accuracy': accuracy_score(y_test, pred_labels), 
        'precision': precision_score(y_test, pred_labels), 
        'recall': recall_score(y_test, pred_labels), 
        'f1': f1_score(y_test, pred_labels), 
        'confusion_matrix': confusion_matrix(y_test, pred_labels).ravel()}
    return evaluation_dict




