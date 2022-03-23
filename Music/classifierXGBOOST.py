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

import seaborn as sbn


def plot_feature_importance(model, name):
    importance_plot = plot_importance(model)
    if name == "total":
        importance_plot.figure.set_size_inches(10,60)
    else:
        importance_plot.figure.set_size_inches(10,20)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    importance_plot.figure.savefig('./figures/importance_plot_' + name + '.png', bbox_inches='tight')
    plt.clf()


def plot_confusion_matrix(y, y_pred, name):
    confmatr    = metrics.confusion_matrix(y, y_pred)
    matrix      = pandas.DataFrame(confmatr, index=["ESC", "SR"], columns=["ESC","SR"])
    plot_matrix = sbn.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plot_matrix.set(xlabel='Predicted', ylabel='True')
    plot_matrix.figure.set_size_inches(5,5)
    plot_matrix.figure.savefig('./figures/confusion_matrix_' + name + '.png')
    plt.clf()


def evaluation_metrics(metrics_d, y, y_pred, y_pred_proba, name):
    accuracy      = metrics.accuracy_score(y, y_pred)
    matt_corrcoef = metrics.matthews_corrcoef(y, y_pred)
    f_score       = metrics.f1_score(y, y_pred, average='weighted')
    cohenkappa    = metrics.cohen_kappa_score(y, y_pred)
    roc_auc       = metrics.roc_auc_score(y, y_pred_proba)

    print("\n" + name + " :")
    print("Accuracy : %0.3f" % accuracy)
    print("Cohen Kappa : %0.3f" % cohenkappa)
    print("F-score : %0.3f" % f_score)
    print("Matthews correlation coefficient : %0.3f" % matt_corrcoef)
    print("ROC AUC : %0.3f" % roc_auc)

    metrics_d['Year']           = metrics_d['Year'] + [name]
    metrics_d['Accuracy']       = metrics_d['Accuracy'] + ["%0.5f" % accuracy]
    metrics_d['MCC']            = metrics_d['MCC'] + ["%0.5f" % matt_corrcoef]
    metrics_d['Cohen\'s Kappa'] = metrics_d['Cohen\'s Kappa'] + ["%0.5f" % cohenkappa]
    metrics_d['F-score']        = metrics_d['F-score'] + ["%0.5f" % f_score]
    metrics_d['ROC AUC']        = metrics_d['ROC AUC'] + ["%0.5f" % roc_auc]

    return metrics_d

def get_most_frequent(tup):
    count = {'ESC' : 0, 'SanRemo' : 0}
    for t in tup:
        count[t] += 1
    return max(count, key=count.get)

plt.style.use('seaborn')

data = pandas.read_csv('./files/features_music_extractor.csv')

data_num = data.select_dtypes(include='number').iloc[:,2:]
data_tot = pandas.concat([data['Contest'], data_num], axis=1)

# Train general classifier
X = data_tot.iloc[:,2:]
y = data_tot.iloc[:,0]

model = XGBClassifier(verbosity=0)
model.fit(X, y, eval_metric='logloss')

# Plot general feature importance
plot_feature_importance(model, "total")

# Cross validation (leave one year out)
dict_year_group = { 2011 : 1, 2012 : 2, 2013 : 3, 2014 : 4, 2015 : 5,
                    2016 : 6, 2017 : 7, 2018 : 8, 2019 : 9, 2021 : 10 }
groups = []
for row in range(0, len(data_tot.index)):
    groups.append(dict_year_group.get(data_tot['Year'][row]))

logo = LeaveOneGroupOut()

scores       = cross_val_score(model, X, y, groups=groups, cv=logo)
y_pred       = cross_val_predict(model, X, y, groups=groups, cv=logo)
y_pred_proba = cross_val_predict(model, X, y, groups=groups, cv=logo, method='predict_proba')

predictions = {'Total' : y_pred}

# Calculating and printing various evaluation metrics
plot_confusion_matrix(y, y_pred, "total")

metrics_d = {'Year': [], 'Accuracy': [], 'MCC': [], 'Cohen\'s Kappa': [], 'F-score': [], 'ROC AUC': []}
metrics_d = evaluation_metrics(metrics_d, y, y_pred, y_pred_proba[:,1], "Total")



# Train classifier per year
for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    data_tot_year = data_tot.loc[data_tot['Year'] == year]

    X_year = data_tot_year.iloc[:,2:]
    y_year = data_tot_year.iloc[:,0]

    model_year = XGBClassifier(verbosity=0)
    model_year.fit(X_year, y_year, eval_metric='logloss')

    # Plot feature importance per year
    plot_feature_importance(model_year, str(year))

    # Cross validation with random groups and ...
    y_pred = {}
    y_pred_proba = {}
    y_pred_year = {}
    y_pred_proba_year = {}
    for i in range(0,5):
        kf = KFold(n_splits=10, shuffle=True)

        # ... the general model
        y_pred_i       = cross_val_predict(model, X_year, y_year, cv=kf)
        y_pred_proba_i = cross_val_predict(model, X_year, y_year, cv=kf, method='predict_proba')

        y_pred[i]       = y_pred_i
        y_pred_proba[i] = y_pred_proba_i

        # ... the yearly model
        y_pred_year_i       = cross_val_predict(model_year, X_year, y_year, cv=kf)
        y_pred_proba_year_i = cross_val_predict(model_year, X_year, y_year, cv=kf, method='predict_proba')

        y_pred_year[i]       = y_pred_year_i
        y_pred_proba_year[i] = y_pred_proba_year_i

    # Get the most frequent classification and average probability
    y_pred_lst = zip(y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4])
    y_pred_lst = [get_most_frequent(t) for t in y_pred_lst]

    y_pred_proba_lst = zip(y_pred_proba[0], y_pred_proba[1], y_pred_proba[2], y_pred_proba[3], y_pred_proba[4])
    y_pred_proba_lst = [(sum(t)/5) for t in y_pred_proba_lst]
    y_pred_proba_list = []
    for i in range(0, len(y_pred_proba_lst)):
        y_pred_proba_list.append(y_pred_proba_lst[i][1])

    y_pred_year_lst = zip(y_pred_year[0], y_pred_year[1], y_pred_year[2], y_pred_year[3], y_pred_year[4])
    y_pred_year_lst = [get_most_frequent(t) for t in y_pred_year_lst]

    y_pred_proba_year_lst = zip(y_pred_proba_year[0], y_pred_proba_year[1], y_pred_proba_year[2], y_pred_proba_year[3], y_pred_proba_year[4])
    y_pred_proba_year_lst = [(sum(t)/5) for t in y_pred_proba_year_lst]
    y_pred_proba_year_list = []
    for i in range(0, len(y_pred_proba_year_lst)):
        y_pred_proba_year_list.append(y_pred_proba_year_lst[i][1])


    # Calculating and printing various evaluation metrics
    predictions[str(year)+'_general'] = y_pred_lst
    predictions[str(year)] = y_pred_year_lst

    plot_confusion_matrix(y_year, y_pred_year_lst, str(year))
    plot_confusion_matrix(y_year, y_pred_lst, str(year) + "_general")

    metrics_d = evaluation_metrics(metrics_d, y_year, y_pred_year_lst, y_pred_proba_year_list, str(year))
    metrics_d = evaluation_metrics(metrics_d, y_year, y_pred_lst, y_pred_proba_list, str(year) + "_general")                               

metrics_df = pandas.DataFrame.from_dict(metrics_d)
print(metrics_df)
print(metrics_df.to_latex(index=False))

print(predictions)



