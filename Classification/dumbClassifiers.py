"""
In dumbClassifiers.py, we implement the world's simplest classifiers:
  1) Always predict +1
  2) Always predict the most frequent label from the training data
  3) Just use the sign of the first feature to decide on label
"""

from binary import *
from numpy  import *
from collections import Counter


import util

class AlwaysPredictOne(BinaryClassifier):
    """
    This defines the classifier that always predicts +1.
    """

    def __init__(self, opts):
        """
        do nothing
        """

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictOne"

    def predict(self, X):
        return 1       # return our constant prediction

    def train(self, X, Y):
        """
        do nothing
        """


class AlwaysPredictMostFrequent(BinaryClassifier):
    """
    This defines the classifier that always predicts the
    most frequent label from the training data.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, assume most frequent class is +1
        """
        self.mostFrequentClass = 1

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictMostFrequent(%d)" % self.mostFrequentClass

    def predict(self, X):
        """
        X is an vector and we want to make a single prediction: Just
        return the most frequent class!
        """
        ### TODO: YOUR CODE HERE
        return self.mostFrequentClass

    def train(self, X, Y):
        ### TODO: YOUR CODE HERE
        res = Counter(Y).most_common(1)[0][0]  
        self.mostFrequentClass = res
   
        
class FirstFeatureClassifier(BinaryClassifier):
    """
    This defines the classifier that always predicts on the basis of
    the first feature only.  In particular, we maintain two
    predictors: one for when the first feature is >0, one for when the
    first feature is <= 0.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, always return 1
        """
        self.classForPos = 1    # what class should we return if X[0] >  0
        self.classForNeg = 1    # what class should we return if X[0] <= 0

    def online(self):
        return False

    def __repr__(self):
        return "FirstFeatureClassifier(%d,%d)" % (self.classForPos, self.classForNeg)

    def predict(self, X):
        """
        check the first feature and make a classification decision based on it
        """
        ### TODO: YOUR CODE HERE
        if X[0] > 0:
            return self.classForPos
        else:
            return self.classForNeg
        

    def train(self, X, Y):
        '''
        just figure out what the most frequent class is for each value of X[:,0] and store it
        '''
        ### TODO: YOUR CODE HERE
        
        #array to store classification for X[0]>0  and X[0]<=0
        pos = []
        neg = []
        #check if the first feature is negative or positive then adds its classification to positive or negative array.
        for i in range(len(X)):
            if X[i,0] > 0:
                pos.append(Y[i])
            else:
                neg.append(Y[i])
        
        
        #check the most frequent classification for X[0]>0 and assign
        self.classForPos =Counter(pos).most_common(1)[0][0]
        #check the most frequent classification for X[0]<=0 and assign
        self.classForNeg =Counter(neg).most_common(1)[0][0]







