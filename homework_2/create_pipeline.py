'''
The goal of this program is to building a simple, modular, extensible machine
learning pipeline. The pipeline has functions that can do the following tasks:

1. Read/Load Data
2. Pre-Process and Clean Data
3. Explore Data
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
    return pd.read_csv(csv_filename)

# Step 2: Explore Data 
# def summary_table_discrete(df, var_name): 
#     return (df[var_name].value_counts(normalize=True)
#                         .to_frame('pct')
#                         .rename_axis(var_name)
#                         .reset_index()) 

# def summary_table_continuous(df, var_name): 
#     return df[var_name].describe()

def explore_boxplot(df, continuous_var, group_by_var, figsize=(20, 5), vert=False): 
    df.boxplot(column=[continuous_var], by=group_by_var, figsize=figsize, vert=vert)

def explore_stackedbar(df, categorical_var, group_by_var, figsize=(20, 5)): 
    table = pd.crosstab(df[categorical_var], df[group_by_var])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=figsize)


# Step 3: Pre-Process/Clean Data 
def process_data(df, index_col=None, na_fill=None, col_types=None): 
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
def continuous_to_categorical(df, categorical_var, continuous_var, num_bins): 
    df[categorical_var] = pd.qcut(df[continuous_var], num_bins, labels=False)
    return df 

def categorical_to_binary(df, binary_var, categorical_var, cutoff_val):
    df[binary_var] = (df[categorical_var] > cutoff_val).astype(int)
    return df  

# Step 5: Build Machine Learning Classifier
def split_data(df, feature_cols, target_col, test_size, seed): 
    x = df[feature_cols]
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed)    
    return x_train, x_test, y_train, y_test

def build_logreg_classifier(seed, solver): 
    return LogisticRegression(random_state=seed, solver='liblinear')
     
def build_tree_classifier(seed, criterion, max_depth, min_samples_leaf): 
    return DecisionTreeClassifier(
        criterion=criterion, random_state=seed, max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf) 

# Step 6: Evaluate Classifier
def evaluate_classifier(classifier, x_train, y_train, x_test, y_test): 
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))





