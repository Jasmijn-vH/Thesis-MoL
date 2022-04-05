import essentia
import essentia.standard as es
import essentia.streaming

import os
import pandas
import re
import numpy as np

import matplotlib.pyplot as plt
from natsort import os_sorted


feats = []

for contst in ['ESC', 'SanRemo', 'MelodiFestivalen']:
    rootdirectory = './audio/' + contst
    for root, dirs, files in os.walk(rootdirectory):
        for song in os_sorted(files):
            # Extract basic information
            year    = os.path.basename(root)
            contest = root.split('/')[-2]

            if contest == 'ESC':
                place   = re.split("_", song)[0]
                country = re.split("_", song)[1]
                songtit = re.split("_", song)[2]
                perform = re.split("_", song)[3]
            else:
                place   = re.split("_", song)[0]
                country = ''
                songtit = re.split("_", song)[1]
                perform = re.split("_", song)[2]

            # Extract the musical features
            features, features_frames = es.MusicExtractor()(os.path.join(root, song))

            # Only erbbands
            re_bark = re.compile('^.*barkbands.*$')
            re_mel  = re.compile('^.*melbands.*$')
            re_mfcc = re.compile('^.*mfcc.*$')
            re_meta = re.compile('^.*metadata.*$')
            
            use_feats = [ f for f in sorted(features.descriptorNames())
                                if not (re_bark.match(f) or re_mel.match(f) or re_mfcc.match(f) or re_meta.match(f))]

            # Construct a dictionary of the features
            dictio = {
                'Contest': contest,
                'Year': year,
                'Country': country,
                'Song': songtit,
                'Performer': perform,
                'Place': place
            }
            for feature_name in use_feats:
                dictio[feature_name] = features[feature_name]
            
            feats.append(dictio)


# Collect all data in a DataFrame and save as .csv
features_dataframe = pandas.DataFrame(feats)
if not os.path.exists('files'):
    os.makedirs('files')
features_dataframe.to_csv('./files/features_music_extractor_orig.csv')


# Encode parts of the data
enc_data = pandas.read_csv('./files/features_music_extractor_orig.csv')

# Encode nominal variables using One Hot Encoding
noms = ['tonal.chords_key', 'tonal.chords_scale', 
        'tonal.key_edma.key', 'tonal.key_edma.scale',
        'tonal.key_krumhansl.key', 'tonal.key_krumhansl.scale',
        'tonal.key_temperley.key', 'tonal.key_temperley.scale']

enc_data = pandas.get_dummies(enc_data, columns=noms)

# Process sGFCC lists
for n in range(0,13):
    enc_data['lowlevel.gfcc.mean_' + str(n)] = np.nan
for row in range(0, len(enc_data.index)):
    row_list = enc_data['lowlevel.gfcc.mean'][row]
    row_list = row_list.replace("[", "").replace("]", "").replace("\n", " ")
    row_list = row_list.split()
    for n in range(0,13):
        enc_data['lowlevel.gfcc.mean_' + str(n)][row] = float(row_list[n])

# Process THPCP
for n in range(0,36):
    enc_data['tonal.thpcp_' + str(n)] = np.nan
for row in range(0, len(enc_data.index)):
    row_list = enc_data['tonal.thpcp'][row]
    row_list = row_list.replace("[", "").replace("]", "").replace("\n", " ")
    row_list = row_list.split()
    for n in range(0,36):
        enc_data['tonal.thpcp_' + str(n)][row] = float(row_list[n])


enc_data = enc_data.iloc[:,1:]

enc_data.to_csv('./files/features_music_extractor.csv')
    
