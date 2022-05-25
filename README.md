# Thesis Master of Logic

This repository contains the implementation for my Master's Thesis.
It was written in Python version 3.8.10.

## How to use
### Audio and Features
To download the audio files, run ```scrapeAudio.py```.
Then, to obtain the audio features, run ```extractFeatures.py```.

### Classification
We constructed three sets of classifiers. 
First ```classifierXGBoost_Italy.py``` contains the classifiers for comparing the Eurovision and Sanremo.
For the Eurovision and Melodifestivalen, see ```classifierXGBoost_Sweden.py```. 
Finally, ```classifierXGBoost_Three.py``` trains the three-class classifier. 

### Predictions
The implementation for the predictions for the general outcome of the Eurovision can be found in ```prediction_Eurovision.py```.
The separate predictions and the comparison to the voting behaviour of all countries are contained in ```votingbehaviour_Italy.py``` (for the Sanremo prediction) and ```votingbehaviour_Sweden.py``` (for the Melodifestivalen prediction).


