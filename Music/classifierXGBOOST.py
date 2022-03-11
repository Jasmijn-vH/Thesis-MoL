import os
import pandas
import numpy as np
import re

from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder



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

# Obtain overall feature importance
X = data_tot.iloc[:,2:]
y = data_tot.iloc[:,0]

model = XGBClassifier()
model.fit(X, y)

importance_plot = plot_importance(model)
importance_plot.figure.set_size_inches(10,60)
if not os.path.exists('figures'):
    os.makedirs('figures')
importance_plot.figure.savefig('./figures/importance_plot_total.png', bbox_inches='tight')


# Obtain feature importance per year
for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    data_tot_year = data_tot.loc[data_tot['Year'] == year]

    X_year = data_tot_year.iloc[:,2:]
    y_year = data_tot_year.iloc[:,0]

    model_year = XGBClassifier()
    model_year.fit(X_year, y_year)

    importance_plot_year = plot_importance(model_year)
    importance_plot_year.figure.set_size_inches(10,20)
    importance_plot_year.figure.savefig('./figures/importance_plot_' + str(year) + '.png', bbox_inches='tight')


# # Test the classifier on 2022 songs
# data_test = pandas.read_csv('./predictions2022/features_music_extractor2022.csv')

# # TO DO: How to handle lists of floats in data?
# #enc_data_test = pandas.get_dummies(data_test, columns=noms)
# data_num_test = data_test.select_dtypes(include='number').iloc[:,1:]
# data_tot_test = pandas.concat([data_test['Contest'], data_num_test], axis=1)

# # Obtain overall feature importance
# X_test = data_tot_test.iloc[:,2:]
# y_test = data_tot_test.iloc[:,0]

# y_pred = model.predict(X_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy : %.2f%%" % (accuracy * 100.0))