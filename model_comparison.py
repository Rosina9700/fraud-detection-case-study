import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

import pandas as pd
from pymongo import MongoClient
import seaborn as sn
from pandas.plotting import scatter_matrix


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import GridSearchCV



def get_data():
    client = MongoClient()
    db = client.fraud_detection
    collection = db.fraud
    df = pd.DataFrame(list(collection.find()))
    return df


def feature_engineering(df):

    df = df[['org_twitter','body_length','user_age','sale_duration2','delivery_method','org_facebook','acct_type']].dropna(axis=0)
    cols = ['body_length','user_age','sale_duration2']

    df['facebook_presence'] = df.org_facebook.apply(lambda x:1 if x>5 else 0)
    df['fraud'] = df['acct_type'].apply(lambda x: True  if 'fraud' in str(x) else False)
    df['twitter_presence'] = df.org_twitter.apply(lambda x:1 if x>5 else 0)
    cols.append('facebook_presence')
    cols.append('twitter_presence')

    delivery_methods = df['delivery_method'].unique()
    for d in delivery_methods[1:]:
        col_name = 'delivery_'+str(d)
        cols.append(col_name)
        df[col_name] = df['delivery_method'].apply(lambda x: 1 if x == d else 0)

    X_train, X_test, y_train, y_test = train_test_split (df[cols],df['fraud'])
    return X_train, X_test, y_train, y_test

def cost_function(labels, predict_probs, threshold=.5):
    cost_benefit = np.array([[2000,-500],[0,0]])
    predict_probs = np.array(predict_probs[:,-1])
    predicted_labels = np.array([0] * len(predict_probs))

    predicted_labels[predict_probs >= threshold] = 1
    cm = standard_confusion_matrix(labels, predicted_labels)
    profit = (cm * cost_benefit).sum() * 1. / len(labels)
    return profit

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D
    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def get_best_estimator(models, X_train, y_train):
    clfs = grid_search(models, X_train, y_train)
    max_score = 0
    best_estimator = None
    for clf in clfs:
        if clf.best_score_ > max_score:
            max_score = clf.best_score_
            best_estimator = clf.best_estimator_
    return best_estimator, max_score, clfs

def grid_search(models, X_train, y_train):
    clfs =[]
    scorer = make_scorer(cost_function, greater_is_better=True, needs_proba=True)

    for key, value in models.items():
        model = GridSearchCV(value[0], cv=5,param_grid=value[1], scoring = scorer)
        clf = model.fit(X_train, y_train)
        clfs.append(clf)
    return clfs

if __name__ == '__main__':
    df = get_data()
    X_train, X_test, y_train, y_test = feature_engineering(df)

    models = {'lr' : [LogisticRegression(),{'n_jobs':[1,-1]}],
            'rfc' : [RandomForestClassifier(n_jobs=-1,random_state=3),{'n_estimators':[40,60,80,120],'max_depth':[30,40,60], 'max_features' : [1,2,3,4]}],
            'knn' : [KNeighborsClassifier(n_jobs=-1),{'n_neighbors': [10,20,30,40],'weights':['uniform','distance']}]}

    best_estimator, best_score, clfs = get_best_estimator(models,X_train,y_train)



    # clfs[0].best_score_
    # clfs[1].best_params_
    # clfs[2].best_params_



    # clfs =[]
    # for key, value in models.items():
    #     model = GridSearchCV(value[0], cv=5,param_grid=value[1], scoring = scorer)
    #     clf = model.fit(X_train, y_train)
    #     clfs.append(clf)
    # return clfs
    #
    # max_score = 0
    # best_estimator = None
    # for clf in clfs:
    #     if clf.best_score_ > max_score:
    #         max_score = clf.best_score_
    #         best_estimator = clf.best_estimator_
    # return best_estimator, max_score
