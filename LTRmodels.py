import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ranking as rnk


class LTRmodel:
    def score(self, docs):
        pass
    def rank(self, scores, docs):
        self.all_rankings, self.all_inverted_rankings = rnk.data_split_rank_and_invert(scores, docs)
        return self.all_rankings, self.all_inverted_rankings
    def loss_function(self, output, target):
        pass

class randomModel(LTRmodel):
    def __init__(self, num_features):
        self.params = np.random.uniform(size=num_features)
    def score(self, docs):
        self.scores = np.dot(docs.feature_matrix, self.params)
        return self.scores
    def loss_function(self, output, target):
        return 1

class pointWiseModel(LTRmodel):
    pass
    ##TODO

class pairWiseModel(LTRmodel):
    pass
    ##TODO

class listWiseModel(LTRmodel):
    pass
    ##TODO
