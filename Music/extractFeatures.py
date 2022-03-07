import essentia
import essentia.standard as es
import essentia.streaming

import os
import pandas
import re

import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory


rootdirectory = './audio'
feats = []

for root, dirs, files in os.walk(rootdirectory):
    for song in files:
        # Extract basic information
        year    = os.path.basename(root)
        contest = root.split('/')[-2]
        country = re.split("_", song)[0]
        songtit = re.split("_", song)[1]
        perform = re.split("_", song)[2]

        # Extract the musical features
        features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                                      rhythmStats=['mean', 'stdev'],
                                                      tonalStats=['mean', 'stdev'])(os.path.join(root, song))
        
        # Construct a dictionary of the features
        dictio = {
            'Contest': contest,
            'Year': year,
            'Country': country,
            'Song': songtit,
            'Performer': perform,
        }
        for feature_name in sorted(features.descriptorNames()):
            dictio[feature_name] = features[feature_name]
        
        feats.append(dictio)

# Collect all data in a DataFrame and save as .csv
features_dataframe = pandas.DataFrame(feats)
if not os.path.exists('files'):
    os.makedirs('files')
features_dataframe.to_csv('./files/features_music_extractor.csv')
    
