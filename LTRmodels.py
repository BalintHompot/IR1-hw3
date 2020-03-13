import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ranking as rnk


class LTRmodel:
    def score(self, docs):
        pass
    def rank(self, docs):
        all_rankings, all_inverted_rankings = rnk.data_split_rank_and_invert(self.score(docs), docs)
        return all_rankings, all_inverted_rankings
    def loss_function(self, output, target):
        pass

class randomModel(LTRmodel):
    def __init__(self, num_features):
        self.params = np.random.uniform(size=num_features)
    def score(self, docs):
        return np.dot(docs.feature_matrix, self.params)
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
