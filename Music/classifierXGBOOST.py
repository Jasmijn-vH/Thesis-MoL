import os
import pandas
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



plt.style.use('ggplot')

data = pandas.read_csv('./files/features_music_extractor.csv')

# TO DO: How to handle categorical data?
data_num = data.select_dtypes(include='number').iloc[:,1:]
data_tot = pandas.concat([data['Contest'], data_num], axis=1)

# Obtain overall feature importance
X = data_tot.iloc[:,2:]
y = data_tot.iloc[:,0]

model = XGBClassifier()
model.fit(X, y)

importance_plot = plot_importance(model)
importance_plot.figure.set_size_inches(10,50)
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

# # TO DO: How to handle categorical data?
# data_num_test = data_test.select_dtypes(include='number').iloc[:,1:]
# data_tot_test = pandas.concat([data_test['Contest'], data_num_test], axis=1)

# # Obtain overall feature importance
# X_test = data_tot_test.iloc[:,2:]
# y_test = data_tot_test.iloc[:,0]

# y_pred = model.predict(X_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy : %.2f%%" % (accuracy * 100.0))