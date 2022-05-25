import os
import pandas
import numpy as np
import re

from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneGroupOut, KFold, cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel

import seaborn as sbn


def plot_feature_importance(model, name):
    importance_plot = plot_importance(model)
    if name == "total":
        importance_plot.figure.set_size_inches(10,60)
    else:
        importance_plot.figure.set_size_inches(10,20)
    importance_plot.figure.savefig('./figures/importance_plot_' + name + '.png', bbox_inches='tight')
    plt.clf()


def plot_confusion_matrix(y, y_pred, name):
    confmatr    = metrics.confusion_matrix(y, y_pred, labels=["ESC", "SanRemo", "MelodiFestivalen"])
    matrix      = pandas.DataFrame(confmatr, index=["ESC", "SR", "MF"], columns=["ESC","SR", "MF"])
    plot_matrix = sbn.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plot_matrix.set(xlabel='Predicted', ylabel='True')
    plot_matrix.figure.set_size_inches(5,5)
    plot_matrix.figure.savefig('./figures/confusion_matrix_' + name + '.png')
    plt.clf()


def evaluation_metrics(metrics_d, y, y_pred, y_pred_proba, scores, name):
    accuracy      = metrics.accuracy_score(y, y_pred)
    matt_corrcoef = metrics.matthews_corrcoef(y, y_pred)
    f_score       = metrics.f1_score(y, y_pred, average='weighted')
    cohenkappa    = metrics.cohen_kappa_score(y, y_pred)

    print("\n" + name + " :")
    print("Accuracy : %0.3f" % accuracy)
    print("Cohen Kappa : %0.3f" % cohenkappa)
    print("F-score : %0.3f" % f_score)
    print("Matthews correlation coefficient : %0.3f" % matt_corrcoef)
    print("Scores :" + str(scores))

    metrics_d['Year']           = metrics_d['Year'] + [name]
    metrics_d['Accuracy']       = metrics_d['Accuracy'] + ["%0.5f" % accuracy]
    metrics_d['MCC']            = metrics_d['MCC'] + ["%0.5f" % matt_corrcoef]
    metrics_d['Cohen\'s Kappa'] = metrics_d['Cohen\'s Kappa'] + ["%0.5f" % cohenkappa]
    metrics_d['F-score']        = metrics_d['F-score'] + ["%0.5f" % f_score]

    return metrics_d



plt.style.use('seaborn')

data = pandas.read_csv('./files/features_music_extractor.csv')

data_num = data.select_dtypes(include='number').iloc[:,1:]
data_all = pandas.concat([data['Contest'], data['Place'], data_num], axis=1)
data_tot = data_all.loc[data_all['Year'] != 2022]

data_tot = data_tot.drop([91, 476, 163, 532, 216, 553, 248, 572, 289, 594, 329, 614, 369, 638])
data_tot = data_tot.drop([2, 689, 43, 721, 98, 753, 126, 785, 161, 817, 205, 845, 247, 873, 291, 901, 332, 929, 382, 957])

# Train general classifier on data from 2011 - 2021
X = data_tot.iloc[:,3:]
y = data_tot.iloc[:,0]
print("Number of features : " + str(len(X.columns)))

model = XGBClassifier(verbosity=0)
model.fit(X, y, eval_metric='logloss')

# Plot general feature importance
plot_feature_importance(model, "total")
importance_plot_top = plot_importance(model, max_num_features=15)
importance_plot_top.figure.savefig('./figures/importance_plot_total_top.png', bbox_inches='tight')
plt.clf()

selection = SelectFromModel(model, threshold=1e-5, prefit=True)
features_select = selection.get_support()
features_select_names = X.columns[features_select]
select_X = selection.transform(X)
print("Number of used features : " + str(len(select_X[0])))

# Cross validation (leave one year out)
dict_year_group = { 2011 : 1, 2012 : 2, 2013 : 3, 2014 : 4, 2015 : 5,
                    2016 : 6, 2017 : 7, 2018 : 8, 2019 : 9, 2021 : 10 }
groups = []
for ind, row in data_tot.iterrows():
    groups.append(dict_year_group.get(data_tot['Year'][ind]))

logo = LeaveOneGroupOut()

scores       = cross_val_score(model, X, y, groups=groups, cv=logo)
y_pred       = cross_val_predict(model, X, y, groups=groups, cv=logo)
y_pred_proba = cross_val_predict(model, X, y, groups=groups, cv=logo, method='predict_proba')

predictions = {'Total' : y_pred}

# Calculating and printing various evaluation metrics
plot_confusion_matrix(y, y_pred, "total")

metrics_d = {'Year': [], 'Accuracy': [], 'MCC': [], 'Cohen\'s Kappa': [], 'F-score': []}
metrics_d = evaluation_metrics(metrics_d, y, y_pred, y_pred_proba[:,1], scores, "Total")