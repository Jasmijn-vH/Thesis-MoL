# Thesis Master of Logic

This repository contains the implementation for my Master's Thesis.
It was written in Python version 3.8.10.

## How to use
### Audio and Features
To download the audio files used in this study, run ```scrapeAudio.py``` (this implementation was partially taken from [1]).
Then, to obtain the audio features, run ```extractFeatures.py```.

### Classification
We constructed three groups of classifiers. 
First ```classifierXGBoost_Italy.py``` contains the classifiers for comparing the Eurovision and Sanremo.
For the Eurovision and Melodifestivalen, see ```classifierXGBoost_Sweden.py```. 
Finally, ```classifierXGBoost_Three.py``` trains the three-class classifier. 

### Predictions
The implementation for the Eurovision predictions based on both national competitions can be found in ```prediction_Eurovision.py```.
The separate predictions and the comparison to the voting behaviour of all countries are contained in ```votingbehaviour_Italy.py``` (for the Sanremo prediction) and ```votingbehaviour_Sweden.py``` (for the Melodifestivalen prediction).


## Files
The folder `files` contains several datafiles that are used as a base in this study.

### Songs
The files ```songsESC.csv```, ```songsMF.csv``` and ```songsSanremo.csv``` contain information about the songs performed at the respective festivals between 2011 and 2021 (with the exception of 2020). 
The first file was partially taken from [1]. For the other files, the official data as published by the Italian broadcaster RAI and the Swedish broadcaster SVT were used.

### Points
The folder `point_allocation_finals` contains an overview of the points distributed in the Eurovision final per year. 
The rows indicate the points allocated to the country corresponding to the row. The columns indicate the points allocated by the country corresponding to the column.
All information for these tables was taken from [2] and [3].


## References
[1] J. Spijkervet. The Eurovision Dataset. https://zenodo.org/badge/latestdoi/214236225, 2020. 

[2] Eurovisionworld. https://eurovisionworld.com/eurovision.

[3] European Broadcasting Union. https://eurovision.tv.
