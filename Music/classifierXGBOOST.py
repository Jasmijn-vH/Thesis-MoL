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
    confmatr = metrics.confusion_matrix(y, y_pred)
    matrix = pandas.DataFrame(confmatr, index=["ESC", "SR"], columns=["ESC","SR"])
    plot_matrix = sbn.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plot_matrix.set(xlabel='Predicted', ylabel='True')
    plot_matrix.figure.set_size_inches(10,10)
    plot_matrix.figure.savefig('./figures/confusion_matrix_' + name + '.png')
    plt.clf()


def evaluation_metrics(metrics_d, y, y_pred, y_pred_proba, y_pred_dummy_mf, y_pred_dummy_st, scores, name):
    matt_corrcoef = metrics.matthews_corrcoef(y, y_pred)
    # f_score_esc = f1_score(y, y_pred, pos_label='ESC')
    # f_score_sr = f1_score(y, y_pred, pos_label='SanRemo')
    fowmal = metrics.fowlkes_mallows_score(y,y_pred)
    cohenkappa_mf = metrics.cohen_kappa_score(y_pred, y_pred_dummy_mf)
    cohenkappa_st = metrics.cohen_kappa_score(y_pred, y_pred_dummy_st)
    roc_auc = metrics.roc_auc_score(y, y_pred_proba[:,1])

    print("\n" + name + " :")
    print("Accuracy : %0.3f, Standard Deviation : %0.3f" % (scores.mean(), scores.std()))
    print("Cohen Kappa (most frequent) : %0.3f" % cohenkappa_mf)
    print("Cohen Kappa (stratified) : %0.3f" % cohenkappa_st)
    # print("F-score (ESC) : %0.3f" % f_score_esc)
    # print("F-score (SR) : %0.3f" % f_score_sr)
    print("Matthews correlation coefficient : %0.3f" % matt_corrcoef)
    print("Fowlkes-Mallows score : %0.3f" % fowmal)
    print("ROC AUC : %0.3f" % roc_auc)

    metrics_d['Year'] = metrics_d['Year'] + [name]
    metrics_d['Accuracy'] = metrics_d['Accuracy'] + ["%0.5f" % scores.mean()]
    metrics_d['Cohen Kappa'] = metrics_d['Cohen Kappa'] + ["%0.5f" % cohenkappa_st]
    metrics_d['MCC'] = metrics_d['MCC'] + ["%0.5f" % matt_corrcoef]
    metrics_d['FM score'] = metrics_d['FM score'] + ["%0.5f" % fowmal]
    metrics_d['ROC AUC'] = metrics_d['ROC AUC'] + ["%0.5f" % roc_auc]

    return metrics_d



plt.style.use('ggplot')

data = pandas.read_csv('./files/features_music_extractor.csv')

# Encode nominal variables using One Hot Encoding
noms = ['tonal.chords_key', 'tonal.chords_scale', 
        'tonal.key_edma.key', 'tonal.key_edma.scale',
        'tonal.key_krumhansl.key', 'tonal.key_krumhansl.scale',
        'tonal.key_temperley.key', 'tonal.key_temperley.scale']

enc_data = pandas.get_dummies(data, columns=noms)

# Process MFCC and GFCC lists
for cc in ['lowlevel.gfcc.mean', 'lowlevel.mfcc.mean']:
    for n in range(0,13):
        enc_data[cc + '_' + str(n)] = np.nan
    for row in range(0, len(enc_data.index)):
        row_list = enc_data[cc][row]
        row_list = row_list.replace("[", "").replace("]", "").replace("\n", " ")
        row_list = row_list.split()
        for n in range(0,13):
            enc_data[cc + '_' + str(n)][row] = float(row_list[n])

# Process THPCP
for n in range(0,36):
    enc_data['tonal.thpcp_' + str(n)] = np.nan
for row in range(0, len(enc_data.index)):
    row_list = enc_data['tonal.thpcp'][row]
    row_list = row_list.replace("[", "").replace("]", "").replace("\n", " ")
    row_list = row_list.split()
    for n in range(0,36):
        enc_data['tonal.thpcp_' + str(n)][row] = float(row_list[n])

data_num = enc_data.select_dtypes(include='number').iloc[:,1:]
data_tot = pandas.concat([enc_data['Contest'], data_num], axis=1)



# Train general classifier
X = data_tot.iloc[:,2:]
y = data_tot.iloc[:,0]

model = XGBClassifier(verbosity=0)
model.fit(X, y, eval_metric='logloss')

# Setting up dummy classifiers
dummy_mf = DummyClassifier(strategy='most_frequent')
dummy_mf.fit(X, y)

dummy_st = DummyClassifier(strategy='stratified')
dummy_st.fit(X, y)

# Plot general feature importance
plot_feature_importance(model, "total")

# Cross validation (leave one year out)
dict_year_group = { 2011 : 1, 2012 : 2, 2013 : 3, 2014 : 4, 2015 : 5,
                    2016 : 6, 2017 : 7, 2018 : 8, 2019 : 9, 2021 : 10 }
groups = []
for row in range(0, len(data_tot.index)):
    groups.append(dict_year_group.get(data_tot['Year'][row]))

logo = LeaveOneGroupOut()

scores = cross_val_score(model, X, y, groups=groups, cv=logo)
y_pred = cross_val_predict(model, X, y, groups=groups, cv=logo)
y_pred_proba = cross_val_predict(model, X, y, groups=groups, cv=logo, method='predict_proba')

y_pred_dummy_mf = cross_val_predict(dummy_mf, X, y, groups=groups, cv=logo)
y_pred_dummy_st = cross_val_predict(dummy_st, X, y, groups=groups, cv=logo)


# Calculating and printing various evaluation metrics
plot_confusion_matrix(y, y_pred, "total")

metrics_d = {'Year': [], 'Accuracy': [], 'Cohen Kappa': [], 'MCC': [], 'FM score': [], 'ROC AUC': []}
metrics_d = evaluation_metrics(metrics_d, y, y_pred, y_pred_proba, y_pred_dummy_mf, y_pred_dummy_st, scores, "Total")


# Train classifier per year
for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    data_tot_year = data_tot.loc[data_tot['Year'] == year]

    X_year = data_tot_year.iloc[:,2:]
    y_year = data_tot_year.iloc[:,0]

    model_year = XGBClassifier(verbosity=0)
    model_year.fit(X_year, y_year, eval_metric='logloss')

    # Setting up dummy classifiers
    dummy_mf_year = DummyClassifier(strategy='most_frequent')
    dummy_mf_year.fit(X_year, y_year)

    dummy_st_year = DummyClassifier(strategy='stratified')
    dummy_st_year.fit(X_year, y_year)

    # Plot feature importance per year
    plot_feature_importance(model_year, str(year))

    # Cross validation with random groups and ...
    kf = KFold(n_splits=10, shuffle=True)

    # ... the general model
    scores = cross_val_score(model, X_year, y_year, cv=kf)
    y_pred = cross_val_predict(model, X_year, y_year, cv=kf)
    y_pred_proba = cross_val_predict(model, X_year, y_year, cv=kf, method='predict_proba')

    y_pred_dummy_mf = cross_val_predict(dummy_mf, X_year, y_year, cv=kf)
    y_pred_dummy_st = cross_val_predict(dummy_st, X_year, y_year, cv=kf)

    # ... the yearly model
    scores_year = cross_val_score(model_year, X_year, y_year, cv=kf)
    y_pred_year = cross_val_predict(model_year, X_year, y_year, cv=kf)
    y_pred_proba_year = cross_val_predict(model_year, X_year, y_year, cv=kf, method='predict_proba')

    y_pred_dummy_mf_year = cross_val_predict(dummy_mf_year, X_year, y_year, cv=kf)
    y_pred_dummy_st_year = cross_val_predict(dummy_st_year, X_year, y_year, cv=kf)

    # Calculating and printing various evaluation metrics
    plot_confusion_matrix(y_year, y_pred_year, str(year))
    plot_confusion_matrix(y_year, y_pred, str(year) + "_general")

    metrics_d = evaluation_metrics(metrics_d, y_year, y_pred_year, y_pred_proba_year, 
                                   y_pred_dummy_mf_year, y_pred_dummy_st_year, scores_year, str(year))
    metrics_d = evaluation_metrics(metrics_d, y_year, y_pred, y_pred_proba, 
                                   y_pred_dummy_mf, y_pred_dummy_st, scores, str(year) + "_general")                               

metrics_df = pandas.DataFrame.from_dict(metrics_d)
print(metrics_df)
print(metrics_df.to_latex(index=False))



