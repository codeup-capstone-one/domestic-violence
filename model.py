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
# cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score




def naive_bayes(feature_list, X_train, y_train):
    fit_frame = X_train[feature_list]
    gnb = GaussianNB().fit(fit_frame, y_train)
    # make predictions with the model
    y_pred = gnb.predict(fit_frame)
    # predict probability with the model
    y_pred_proba = gnb.predict_proba(fit_frame)
    return gnb, y_pred, y_pred_proba


def log_reg(feature_list, X_train, y_train, cv_num=5):
    clf = LogisticRegressionCV(cv=cv_num,
                               random_state=0,
                               ).fit(X_train[feature_list], y_train)
    y_pred = clf.predict(X_train[feature_list])
    y_pred_proba = clf.predict_proba(X_train[feature_list])
    return clf, y_pred, y_pred_proba


def decision_tree(params,
                  feature_list,
                  X_train,
                  y_train,
                  criterion='entropy',
                  max_depth=4,
                  max_features=3):
    dtc = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, max_features=max_features, random_state=0)
    grid = GridSearchCV(dtc, params, cv=3, iid=True)
    grid.fit(X_train[feature_list], y_train)
    results = grid.cv_results_
    results = grid.cv_results_
    test_scores = results['mean_test_score']
    params = results['params']
    for p, s in zip(params, test_scores):
        p['score'] = s
    dtc.fit(X_train[feature_list], y_train)
    print('Cross Validation Results: ')
    print(cross_val_score(dtc, X_train[feature_list], y_train, cv=5))
    y_pred = dtc.predict(X_train[feature_list])
    y_pred_proba = dtc.predict_proba(X_train[feature_list])
    return dtc, y_pred, y_pred_proba
