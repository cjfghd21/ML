"""
Implementation of k-nearest-neighbor classifier
"""

from numpy import *
from pylab import *

from binary import *


class KNN(BinaryClassifier):
    """
    This class defines a nearest neighbor classifier, that support
    _both_ K-nearest neighbors _and_ epsilon ball neighbors.
    """

    def __init__(self, opts):
        """
        Initialize the classifier.  There's actually basically nothing
        to do here since nearest neighbors do not really train.
        """

        # remember the options
        self.opts = opts

        # just call reset
        self.reset()

    def reset(self):
        self.trX = zeros((0,0))    # where we will store the training examples
        self.trY = zeros((0))      # where we will store the training labels

    def online(self):
        """
        We're not online
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the 'vote' in favor of a positive
        or negative label.  In particular, if, in our neighbor set,
        there are 5 positive training examples and 2 negative
        examples, we return 5-2=3.

        Everything should be in terms of _Euclidean distance_, NOT
        squared Euclidean distance or anything more exotic.
        """
            
        isKNN = self.opts['isKNN']     # true for KNN, false for epsilon balls
        N     = self.trX.shape[0]      # number of training examples

        if self.trY.size == 0:
            return 0                   # if we haven't trained yet, return 0
        elif isKNN:
            # this is a K nearest neighbor model
            # hint: look at the 'argsort' function in numpy
            K = self.opts['K']         # how many NN to use

            val = 0                    # this is our return value: #pos - #neg of the K nearest neighbors of X
            ### TODO: YOUR CODE HERE
            dist = sqrt(((X - self.trX) ** 2).sum(axis=1)) #euclidean distance
            sorted_dist = argsort(dist)   #sorted indices 
            # sorted_dist[0:K] gives K indices of dist in closest to furthest order.
            # get values of K cloest neighbors then find how many pos, negative label
            neg = 0
            pos = 0
            for i  in self.trY[sorted_dist[0:K]]:
                if i >=0:
                    pos+=1
                else:
                    neg+=1

            val = pos - neg
            return val
        else:
            # this is an epsilon ball model
            eps = self.opts['eps']     # how big is our epsilon ball

            val = 0                    # this is our return value: #pos - #neg within and epsilon ball of X
            ### TODO: YOUR CODE HERE
            dist = sqrt(((X - self.trX) ** 2).sum(axis=1))
            #track positive and negative values
            neg = 0
            pos = 0
            #list of label within the distance
            within_dist = self.trY[dist <= eps] 

            #check how many positive and negatives for those within dist
            for i  in within_dist:
                if i >=0:
                    pos+=1
                else:
                    neg+=1
            

            val = pos - neg
            return val
                
            


    def getRepresentation(self):
        """
        Return the weights
        """
        return (self.trX, self.trY)

    def train(self, X, Y):
        """
        Just store the data.
        """
        self.trX = X
        self.trY = Y
        
