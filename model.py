# ===========
# ENVIRONMENT
# ===========

import pandas as pd
import numpy as np
# sklearn machine learning
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
# NB:
from sklearn.naive_bayes import GaussianNB
# Decision Tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
# cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# SVC:
from sklearn.svm import SVC


def naive_bayes(feature_list, X_train, y_train):
    '''
    Creates a Naive Bayes model based on
    feature_list: a list of features,
    X_train: a pandas frame of features,
    and
    y_train:  a pandas frame of targets

    returns predictions and probabilities in addition to the model itself.
    '''
    fit_frame = X_train[feature_list]
    gnb = GaussianNB().fit(fit_frame, y_train)
    # make predictions with the model
    y_pred = gnb.predict(fit_frame)
    # predict probability with the model
    y_pred_proba = gnb.predict_proba(fit_frame)
    return gnb, y_pred, y_pred_proba


def log_reg(feature_list, X_train, y_train, cv_num=5, solver='lbfgs'):
    '''
    Creates a Logistic Regression model based on
    feature_list: a list of features,
    X_train: a pandas frame of features,
    and
    y_train:  a pandas frame of targets
    kwargs:
    cv_num: integer for number of cross-validation folds, default to 5.
    solver: hyperparameter for logistic regression, default to lbfgs.
    (Refer to sklearn documentation for further information)

    returns predictions and probabilities of outcome
    prints cross validation results of each model
    '''
    clf = LogisticRegressionCV(cv=cv_num,
                               random_state=0,
                               solver=solver).fit(X_train[feature_list], y_train)
    print('Cross Validation Results: ')
    print(cross_val_score(clf, X_train[feature_list], y_train, cv=cv_num))
    y_pred = clf.predict(X_train[feature_list])
    y_pred_proba = clf.predict_proba(X_train[feature_list])
    return clf, y_pred, y_pred_proba


def decision_tree(feature_list,
                  X_train,
                  y_train,
                  params={'max_depth': [2, 3, 4],
                          'max_features': [None, 1, 3]},
                  criterion='entropy',
                  max_depth=4,
                  max_features=3):
    '''
    Creates a Decision Tree model based on
    feature_list: a list of features,
    X_train: a pandas frame of features,
    and
    y_train:  a pandas frame of targets
    kwargs:
    params: a dictionary of depth and feature hyperparameters for grid cross-validation.
    default depths [2, 3, 4] default max_features [0, 1, 3]
    criterion (default entropy),
    max_depth (default 4), and
    max_feature (default 3) hyperparameters
    (see sklearn documentation for more information)

    Returns a decision tree model in addition to printing predictions and probabilities
    '''
    dtc = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, max_features=max_features, random_state=0)
    grid = GridSearchCV(dtc, params, cv=3, iid=True)
    grid.fit(X_train[feature_list], y_train)
    results = grid.cv_results_
    test_scores = results['mean_test_score']
    params = results['params']
    for p, s in zip(params, test_scores):
        p['score'] = s
    dtc.fit(X_train[feature_list], y_train)
    print('grid cv results: ')
    print(pd.DataFrame(params).sort_values(by='score'))
    print('Cross Validation Results: ')
    print(cross_val_score(dtc, X_train[feature_list], y_train, cv=5))
    y_pred = dtc.predict(X_train[feature_list])
    y_pred_proba = dtc.predict_proba(X_train[feature_list])
    return dtc, y_pred, y_pred_proba


def plot_decision_tree(clf, feature_name, target_name):
    '''
    This function creates a visualization of a decision tree in png format.
    Takes in:
    clf: a decision tree object,
    feature_name: feature name of the training set and a
    target_name: a target variable name (list formatted)'''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


def random_forest(feature_list,
                  X_train,
                  y_train,
                  cv_num=5,
                  bootstrap=True,
                  class_weight=None,
                  criterion='entropy',
                  min_samples_leaf=3,
                  n_estimators=100,
                  max_depth=3,
                  random_state=0,
                  r_params={'max_depth': [2, 3, 4]}):
     '''
    Creates a Decision Tree model based on
    feature_list: a list of features,
    X_train: a pandas frame of features,
    and
    y_train:  a pandas frame of targets
    kwargs:
    cv_num: integer for number of cross-validation folds, default to 5.
    boot_strap: (default True) (see sklearn documentation),
    class_weight: (default None) (see sklearn documentation),
    criterion: (default entropy) (see sklearn documentation),
    min_samples_leaf: (default 3)(see sklearn documentation),
    n_estimators: (default 100), (see sklearn documentation),
    max_depth: (default 3), and
    random_state: (default 0)
    r_params: a dictionary of depth hyperparameters for grid cross-validation. (default depths [2, 3, 4])
    (see sklearn documentation for more information)

    Returns a decision tree model in addition to printing predictions and probabilities
    '''
    rf = RandomForestClassifier(bootstrap=bootstrap,
                                class_weight=class_weight,
                                criterion=criterion,
                                min_samples_leaf=min_samples_leaf,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state)
    r_grid = GridSearchCV(rf, r_params, cv=3, iid=True)
    r_grid.fit(X_train[feature_list], y_train)
    r_results = r_grid.cv_results_
    r_test_scores = r_results['mean_test_score']
    r_params = r_results['params']
    for p, s in zip(r_params, r_test_scores):
        p['score'] = s
    rf.fit(X_train[feature_list], y_train)
    cross_val_score(rf, X_train[feature_list], y_train, cv=5)
    y_pred = rf.predict(X_train[feature_list])
    y_pred_proba = rf.predict_proba(X_train[feature_list])
    print('grid cv results: ')
    print(pd.DataFrame(r_params).sort_values(by='score'))
    print('Cross Validation Results: ')
    print(cross_val_score(rf, X_train[feature_list], y_train, cv=5))
    return rf, y_pred, y_pred_proba
