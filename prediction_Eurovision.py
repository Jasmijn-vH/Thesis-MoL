import os
import pandas
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sbn

from xgboost import XGBRanker, XGBClassifier

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_predict

from scipy.stats import rankdata, spearmanr


def plot_confusion_matrix(y, y_pred, name):
    confmatr    = metrics.confusion_matrix(y, y_pred)
    matrix      = pandas.DataFrame(confmatr, index=["No", "Yes"], columns=["No","Yes"])
    plot_matrix = sbn.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plot_matrix.set(xlabel='Predicted '+ name, ylabel='True ' + name)
    plot_matrix.figure.set_size_inches(5,5)
    plot_matrix.figure.savefig('./figures/Predictions/confusion_matrix_' + name + '.png')
    plt.clf()


def evaluation_metrics(y, y_pred, name):
    accuracy      = metrics.accuracy_score(y, y_pred)
    matt_corrcoef = metrics.matthews_corrcoef(y, y_pred)
    f_score       = metrics.f1_score(y, y_pred, average='weighted')
    cohenkappa    = metrics.cohen_kappa_score(y, y_pred)

    print("\n" + name + " :")
    print("Accuracy : %0.3f" % accuracy)
    print("Cohen Kappa : %0.3f" % cohenkappa)
    print("F-score : %0.3f" % f_score)
    print("Matthews correlation coefficient : %0.3f" % matt_corrcoef)



plt.style.use('seaborn')


data = pandas.read_csv('./files/features_music_extractor.csv')

data_num = data.select_dtypes(include='number').iloc[:,1:]
data_all = pandas.concat([data['Contest'], data['Country'], data['Place'], data_num], axis=1)
data_tot = data_all.loc[data_all['Year'] != 2022]

data_tot_ESC  = data_tot.loc[data_tot['Contest'] == 'ESC']
data_tot_ESC  = data_tot_ESC.loc[data_tot_ESC['Country'] != 'Italy']
data_tot_ESC  = data_tot_ESC.loc[data_tot_ESC['Country'] != 'Sweden']

data_tot_SRMF = data_tot.loc[data_tot['Contest'] != 'ESC']


# Prediction by using a top 10 classifier
for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    data_tot_year_ESC  = data_tot_ESC.loc[data_tot_ESC['Year'] == year]
    data_tot_year_SRMF = data_tot_SRMF.loc[data_tot_SRMF['Year'] == year]

    X_year = data_tot_year_SRMF.iloc[:,4:]
    y_year = data_tot_year_SRMF.iloc[:,2]
    for i in y_year:
        if i in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            y_year = y_year.replace(i, 'Yes')
        else:
            y_year = y_year.replace(i, 'No')

    # Train model on outcomes of SR and MF
    model = XGBClassifier(verbosity=0)
    model.fit(X_year, y_year, eval_metric='logloss')

    # Cross validate it to determine performance
    kf      = KFold(n_splits=10, shuffle=True)
    y_pred  = cross_val_predict(model, X_year, y_year, cv=kf)
    evaluation_metrics(y_year, y_pred, str(year)+'_crossval')

    # Test model to predict top 10 of ESC
    X_test = data_tot_year_ESC.iloc[:,4:]
    y_test = data_tot_year_ESC.iloc[:,2]
    for i in y_test:
        y_test = y_test.replace(i, float(i))
    y_test = rankdata(y_test)

    classes = []
    for i in y_test:
        if i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            classes.append('Yes')
        else:
            classes.append('No')

    predictions = model.predict(X_test)
    probs       = model.predict_proba(X_test)

    yes_probs = list(list(zip(*probs))[1])
    ind = np.argpartition(yes_probs, -10)[-10:]

    predictions_10 = []
    for i in range(0, len(predictions)):
        if i in ind:
            predictions_10 = predictions_10 + ['Yes']
        else:
            predictions_10 = predictions_10 + ['No']

    plot_confusion_matrix(classes, predictions_10, str(year) + '_top10')
    evaluation_metrics(classes, predictions_10, str(year) + '_top10')
    print(predictions_10)


# Prediction by ranker
for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    data_tot_year_ESC  = data_tot_ESC.loc[data_tot_ESC['Year'] == year]
    data_tot_year_SRMF = data_tot_SRMF.loc[data_tot_SRMF['Year'] == year]

    data_tot_year_SRMF = data_tot_year_SRMF.dropna(subset='Place')
    data_tot_year_SRMF = data_tot_year_SRMF.replace(['E', 'QDSQ'], '20')

    # Train ranker
    X = data_tot_year_SRMF.iloc[:,4:]
    y = data_tot_year_SRMF.iloc[:,2]

    group_sizes = data_tot_year_SRMF.groupby(['Contest']).size().to_frame('size')['size'].to_numpy()
    groups = [*group_sizes[1:], *group_sizes[:1]]

    model = XGBRanker()
    model.fit(X, y, group=groups)

    # Predict ranking of ESC
    X_test = data_tot_year_ESC.iloc[:,4:]
    y_test = data_tot_year_ESC.iloc[:,2]
    for i in y_test:
        y_test = y_test.replace(i, float(i))
    y_test = rankdata(y_test)

    predictions  = model.predict(X_test)

    ranks = rankdata(predictions)
    spearman = spearmanr(ranks, y_test)
    print("\n" + str(year))
    print(spearman)
    
