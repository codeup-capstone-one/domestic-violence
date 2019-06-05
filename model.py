import pandas as pd
import numpy as np
# sklearn machine learning
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
# NB:
from sklearn.naive_bayes import GaussianNB

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
    return y_pred, y_pred_proba