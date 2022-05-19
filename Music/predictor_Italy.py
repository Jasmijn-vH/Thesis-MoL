import os
import pandas
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sbn

from xgboost import XGBRanker

from scipy.stats import rankdata, pearsonr


plt.style.use('seaborn')


data = pandas.read_csv('./files/features_music_extractor.csv')

data_num     = data.select_dtypes(include='number').iloc[:,1:]
data_all     = pandas.concat([data['Contest'], data['Country'], data['Place'], data_num], axis=1)
data_all     = data_all.loc[data_all['Year'] != 2022]
data_tot_ESC = data_all.loc[data_all['Contest'] == 'ESC']
data_tot_SR  = data_all.loc[data_all['Contest'] == 'SanRemo']
data_tot     = pandas.concat([data_tot_ESC, data_tot_SR])

pred_rankings_fin     = {}

# COMPUTE THE ITALIAN PREDICTED RANKING PER YEAR
for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    data_tot_year_SR   = data_tot_SR.loc[data_tot['Year'] == year]
    data_tot_year_ESC  = data_tot_ESC.loc[data_tot['Year'] == year]

    data_tot_year_SR = data_tot_year_SR.replace([np.nan], '12')
    data_tot_year_SR = data_tot_year_SR.replace(['E', 'QDSQ', np.nan], '21')

    # Train ranker on data from Sanremo
    X = data_tot_year_SR.iloc[:,4:]
    y = data_tot_year_SR.iloc[:,2]

    group_sizes = data_tot_year_SR.groupby(['Contest']).size().to_frame('size')['size'].to_numpy()
    groups = [*group_sizes[1:], *group_sizes[:1]]

    model = XGBRanker()
    model.fit(X, y, group=groups)

    # Predict ranking of ESC
    X_test = data_tot_year_ESC.iloc[:,4:]
    y_test = data_tot_year_ESC.iloc[:,2]
    for i in y_test:
        y_test = y_test.replace(i, float(i))
    predictions  = model.predict(X_test)

    ranks = rankdata(predictions)

    # Only consider the predicted and actual ranks for countries competing in the final
    if year == 2011:
        ranks_fin = rankdata(ranks[:25])
        act_fin   = y_test[:25]
    else: 
        if year == 2015:
            ranks_fin = rankdata(ranks[:27])
            act_fin   = y_test[:27]
        else:
            ranks_fin = rankdata(ranks[:26])
            act_fin   = y_test[:26]

    pred_rankings_fin[year] = ranks_fin

    # Compute correlation with actual results from final
    pred_rank_fin     = pandas.DataFrame(pred_rankings_fin[year], index=data_tot_year_ESC['Country'][:len(pred_rankings_fin[year])])
    pred_rank_fin     = rankdata(pred_rank_fin.drop('Italy', errors='ignore'))
    act_fin_df        = pandas.DataFrame(list(act_fin), index=data_tot_year_ESC['Country'][:len(act_fin)])
    act_fin_df        = rankdata(act_fin_df.drop('Italy', errors='ignore'))

    pearson_year = pearsonr(pred_rank_fin, act_fin_df)
    print(str(year) + " : " + str(pearson_year))


# COMPARE THE ITALIAN PREDICTION WITH THE ACTUAL VOTING OF ALL COUNTRIES
result_df = pandas.DataFrame()

for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021]:
    # Load the point allocation and drop all points awarded to Italy
    point_alloc_year = pandas.read_csv('./files/point_allocation_finals/' + str(year) + '.csv', sep=';', index_col='Country')
    point_alloc_year = point_alloc_year.replace(np.nan, 0)
    point_alloc_year = point_alloc_year.drop('Italy', errors='ignore')

    data_tot_year_ESC = data_tot_ESC.loc[data_tot['Year'] == year]
    countries_year    = point_alloc_year.columns.tolist()[3:]

    # Retrieve the predicted ranking by Italy and drop Italy
    pred_rank_fin     = pandas.DataFrame(pred_rankings_fin[year], index=data_tot_year_ESC['Country'][:len(pred_rankings_fin[year])])
    pred_rank_fin     = pred_rank_fin.drop('Italy', errors='ignore')
    
    coefs = []
    pvals = []

    for country in countries_year:
        # Rank the points awarded by a specific country, don't consider that particular country (one cannot vote for themselves)
        point_alloc_country = point_alloc_year[country].drop(country, errors='ignore')
        point_rank_country  = rankdata([-1 * i for i in point_alloc_country], method='average')

        # Revise the predicted Italian ranking by dropping the current country
        pred_rank_fin_country = rankdata(pred_rank_fin.drop(country, errors='ignore'))      

        # Compare the predicted Italian ranking and the actual ranking from the current country
        # By computing Pearson's correlation coefficient
        pearson_country = pearsonr(pred_rank_fin_country, point_rank_country)
        coefs.append("%0.2f" % pearson_country[0])
        pvals.append("%0.2f" % pearson_country[1])

    df_pears_year = pandas.DataFrame([coefs, pvals], index=[str(year)+'_coef', str(year)+'_pval'], columns=countries_year)

    result_df = result_df.append(df_pears_year)

result_df = result_df.reindex(sorted(result_df.columns), axis=1) 
result_df = result_df.replace(np.nan, '')   
print(result_df)

# Automatically construct latex tables
results_1114 = result_df.loc[:'2014_pval']
results_1517 = result_df.loc['2015_coef':'2017_pval']
results_1821 = result_df.loc['2018_coef':'2021_pval']

result_1114_transposed = results_1114.transpose() 
result_1517_transposed = results_1517.transpose() 
result_1821_transposed = results_1821.transpose() 

print(result_1114_transposed.to_latex(index=True))
print(result_1517_transposed.to_latex(index=True))
print(result_1821_transposed.to_latex(index=True))

