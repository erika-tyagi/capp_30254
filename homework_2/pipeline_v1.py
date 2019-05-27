'''
The goal of this program is to building a simple, modular, extensible machine
learning pipeline. The pipeline has functions that can do the following tasks:

1. Read/Load Data
2. Explore Data (separately in write-up)
3. Pre-Process and Clean Data
4. Generate Features/Predictors
5. Build Machine Learning Classifier
6. Evaluate Classifier 
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# Step 1: Read/Load Data 
def load_data(csv_filename): 
    '''
    Inputs: csv_filename
    Returns: pandas df 
    '''
    return pd.read_csv(csv_filename)


# Step 3: Pre-Process/Clean Data 
def process_data(df, index_col=None, na_fill=None, col_types=None): 
    '''
    Sets the dataframe's index, replaces missing values, and specifies column data types. 
    Inputs: 
    - df: pandas df
    - index_col: column name of the index 
    - na_fill: specifies how to replace missing values: 
        - default: drop rows with missing values
        - 'mean': replace with column mean
        - 'median': replace with column median 
    - col_types: dictionary specifying column data types 
    Returns: pandas df 
    '''
    if index_col:
        df = df.set_index(index_col)

    if not na_fill: 
        df = df.dropna()
    elif na_fill == 'mean': 
        df = df.fillna(df.mean())
    elif na_fill == 'median': 
        df = df.fillna(df.median())

    if col_types: 
        df = df.astype(col_types)

    return df 


# Step 4: Generate Features/Predictors
def continuous_to_categorical(df, categorical_col, continuous_col, num_bins): 
    '''
    Creates a new categorical variable from an existing continuous variable. 
    Inputs: 
    - df: pandas df
    - categorical_col: the new column name
    - continuous_col: the existing column name
    - num_bins: the number of bins to split the continuous variable into 
    Returns: pandas df
    '''
    df[categorical_col] = pd.qcut(df[continuous_col], num_bins, labels=False)
    return df 

def categorical_to_binary(df, binary_col, categorical_col, cutoff_val):
    '''
    Creates a new binary variable from an existing continuous variable. 
    Inputs: 
    - df: pandas df
    - binary_col: the new column name
    - categorical_col: the existing column name
    - cutoff_val: the cutoff value to determine the binary split 
    Returns: pandas df 
    '''
    df[binary_col] = (df[categorical_col] > cutoff_val).astype(int)
    return df  


# Step 5: Build Machine Learning Classifier
def split_data(df, feature_cols, target_col, test_size, seed): 
    '''
    Splits the data into training and testing dataframes. 
    Inputs: 
    - df: pandas df
    - feature_cols: list of column names to include as features
    - target_col: target columnn name 
    - test_size: the proportion of data to use for testing 
    - seed: random seed
    Returns: 4 dfs (features training, features testing, target training, target testing)
    '''
    x = df[feature_cols]
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed)    
    return x_train, x_test, y_train, y_test

def build_logreg_classifier(seed, solver='liblinear'): 
    '''
    Creates a logistic regression classifier 
    Inputs: 
    - seed: random seed  
    - solver: algorithm to use in the optimization problem 
    Returns: sklearn Logistic Regression object 
    '''
    return LogisticRegression(random_state=seed, solver=solver)
     
def build_tree_classifier(seed, criterion, max_depth, min_samples_leaf): 
    '''
    Creates a decision tree classifier 
    Inputs: 
    - seed: random seed  
    - criterion: the function to measure the quality of a split (gini, entropy)
    - max_depth: the tree's maximum depth 
    - min_samples_leaf: the minumum number of samples required to split a node
    Returns: sklearn Decision Tree object 
    '''
    return DecisionTreeClassifier(
        criterion=criterion, random_state=seed, max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf) 


# Step 6: Evaluate Classifier
def evaluate_classifier(classifier, x_train, y_train, x_test, y_test): 
    '''
    Fits the classifier based on the training data, classifies the testing data, 
    and reports an accuracy score. 
    Inputs: 
    - classifier: sklearn classifier object 
    - x_train, y_train, x_test, y_test: created in split_data 
    Returns: None (prints accuracy score)
    '''
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))





