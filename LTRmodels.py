import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ranking as rnk
DEVICE = "cpu"

## parent class of LTR model with shallow MLP as scoring
class LTRmodel(nn.Module):
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3):
        super().__init__()
        self.scoringModules = nn.ModuleList()
        input_size = num_features
        for layer_size in scoring_network_layers:
            self.scoringModules.append(nn.Sequential(nn.Linear(input_size, layer_size), nn.Dropout(dropout), nn.ReLU()))
            input_size = layer_size
        ## final scoring layer, mapping to a single number
        self.scoringModules.append(nn.Linear(layer_size, 1))
        self.scoringModules.to(DEVICE)
        self.name = "Parent LTR model class"
        self.loss_fn = None

    def score(self, docs):
        scores = torch.Tensor(docs.feature_matrix)
        scores = scores.to(DEVICE)
        for layer in self.scoringModules:
            scores = layer(scores)
        self.scores = scores
        return scores.detach().numpy().squeeze()
    def rank(self, scores, docs):
        self.all_rankings, self.all_inverted_rankings = rnk.data_split_rank_and_invert(scores, docs)
        return self.all_rankings, self.all_inverted_rankings
    def loss_function(self, target):
        return self.loss_fn(self.scores, target)


class pointWiseModel(LTRmodel):
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3):
        super().__init__(num_features, scoring_network_layers, dropout)
        self.name = "Pointwise LTR model"
        self.loss_fn = torch.nn.SmoothL1Loss()

class pairWiseModel(LTRmodel):
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 1000):
        super().__init__(num_features, scoring_network_layers, dropout)
        self.name = "Pairwise LTR model"
        self.loss_fn = torch.nn.BCELoss()
        self.sigma = sigma
        self.random_pairs = random_pairs

    def calc_p(self,s_i, s_j):
        return 1/(1 + torch.exp(-self.sigma * (s_i - s_j)))

    #override
    def loss_function(self,target):
        target = target.squeeze()
        loss = torch.zeros(1)
        

        ## generate pairs and calculate P and S
        for pair in range(self.random_pairs):
            i, j = np.random.choice(self.scores.shape[0], 2)
            s_i = self.scores[i]
            s_j = self.scores[j]
            if target[i] > target[j]:
                S = 1
            elif target[i] == target[j]:
                S = 0
            else:
                S = -1
            
            S = torch.Tensor([S])
            P = self.calc_p(s_i, s_j)
            loss += self.loss_fn(P, S)

        return loss

class listWiseModel(LTRmodel):
    pass
    ##TODO
