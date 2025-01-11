#%%
from numpy import *
from pylab import *

import dumbClassifiers
import datasets
import util
import runClassifier
import dt
import knn
import perceptron

#%%
h = dt.DT({'maxDepth': 1})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

# %%
h = dt.DT({'maxDepth': 2})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
h
# %%
h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
h


# %%
h = dt.DT({'maxDepth': 2})
h.train(datasets.SentimentData.X, datasets.SentimentData.Y)
h

# %%
print(datasets.SentimentData.words[626] + "\n")
print(datasets.SentimentData.words[683] + "\n")
print(datasets.SentimentData.words[1139] + "\n")

# %%
runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.SentimentData)
# %%
runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.SentimentData)
# %%
runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)
# %%
curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 9}), datasets.SentimentData)
# %%
runClassifier.plotCurve('DT on Sentiment Data', curve)
# %%
curve = runClassifier.hyperparamCurveSet(dt.DT({}), 'maxDepth', [1,2,4,6,8,12,16], datasets.SentimentData)
#%%
runClassifier.plotCurve('DT on Sentiment Data (hyperparameter)', curve)
#%%
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
# %%
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)

# %%
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
# %%
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
#%%
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)

# %%
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
# %%
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)

#%%
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6.0}), datasets.DigitData)
#%%
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 10.0}), datasets.DigitData)
# %%
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.TennisData)
# %%
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData)
#%%
runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptron.Perceptron({'numEpoch': 200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h
runClassifier.plotClassifier(array([ 7.3, 18.9]), 0.0)
# %%
