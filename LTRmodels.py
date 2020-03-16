import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ranking as rnk
from tqdm import tqdm
from evaluate import dcg_at_k, ndcg_at_k #for lamdaRank
DEVICE = "cpu"

## parent class of LTR model with shallow MLP as scoring
class LTRmodel(nn.Module):
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3):
        super().__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.scoringModules = self.constructScoringNetwork(scoring_network_layers)
        self.name = "Parent LTR model class"
        self.loss_fn = None

    def constructScoringNetwork(self, scoring_network_layers):
        input_size = self.num_features
        scoringModules = nn.ModuleList()
        for layer_size in scoring_network_layers:
            scoringModules.append(nn.Sequential(nn.Linear(input_size, layer_size), nn.Dropout(self.dropout), nn.ReLU()))
            input_size = layer_size
        ## final scoring layer, mapping to a single number
        scoringModules.append(nn.Linear(layer_size, 1))
        return scoringModules.to(DEVICE)

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
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 500):
        super().__init__(num_features, scoring_network_layers, dropout)
        self.name = "Pointwise LTR model"
        self.loss_fn = torch.nn.SmoothL1Loss()

class RankNetDefualt(LTRmodel):
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 500):
        super().__init__(num_features, scoring_network_layers, dropout)
        self.name = "Ranknet-original"
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

class RankNetFast(LTRmodel):
    """
    Factorisation approach to RankNet pairwise approach
    """
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 500):
        super().__init__(num_features, scoring_network_layers, dropout)
        self.name = "RankNetFast"
        # self.loss_fn = torch.nn.BCELoss()
        self.sigma = sigma
        self.random_pairs = random_pairs

    def calc_p(self,s_i, s_j):
        return 1/(1 + torch.exp(-self.sigma * (s_i - s_j)))

    def loss_function(self,target):
        """loss function (overriding that of parent class LTRmodel)

        :arg1: target
        :returns: sum_i(lambda_ i) x sum_i(s_i); the gradient of the returned is akin to d(loss)/d(parameters)
                    see equ (5) of the assignment sheet
                    lambda is detached from backprop; s_i retains gradient
                    lambda acts to scale the backpropogation wrt. sum(s_i)
        """

        target = target.squeeze() #relevancy labels 0 to 4 for each document
        loss = torch.zeros(1)

        lambda_i = 0
        ## generate pairs and calculate P and S


        #pick documents at random for optimisation (enough documents for 500 comparison operations)
        n = int(np.ceil(np.sqrt(self.random_pairs))) #i.e. sqrt(500) ~= 23
        random_docs = np.random.choice(self.scores.shape[0], n) #i.e. 23*22 ~= 500 random pairs

        #calculate lambda_i and s_i, then iterate loss accordingly
        for i in random_docs:
            for j in random_docs:
                if i == j: continue

                s_i = self.scores[i]
                s_j = self.scores[j]

                if target[i] > target[j]:
                    S = 1
                elif target[i] == target[j]:
                    S = 0
                else:
                    S = -1

                S = torch.Tensor([S])
                z_i = s_i.detach() ; z_j = s_i.detach() #detach scores for lambda calc
                lambda_ij = float(self.sigma*(0.5*(1-S) - self.calc_p(z_i, z_j))[0]) #lecture notes equa 34
                # print(lambda_ij)

                lambda_i += lambda_ij
            # print(s_i, s_j, self.calc_p(s_i, s_j), lambda_ij, lambda_i)
            loss += lambda_i*s_i

        return loss

class listWiseModel(LTRmodel):
    """LambdaRank implementation
    """
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 500):
        super().__init__(num_features, scoring_network_layers, dropout)
        self.name = "LambdaRank"
        # self.loss_fn = torch.nn.BCELoss()
        self.sigma = sigma
        self.random_pairs = random_pairs

    def calc_p(self,s_i, s_j):
        return 1/(1 + torch.exp(-self.sigma * (s_i - s_j)))

    def loss_function(self,target, irm = "ERR"):
        """loss function (overriding that of parent class LTRmodel)

           gradient for backprop: dC/d(params) = dC/d(s_i) . d(si)/d(params)
           dC/d(s_i) = lambda_i . |delta irm(ij)|  - note1: irm(ij) is the change in irm from swapping docs ; note2: gradients disconnected for this

           Thus, the 'pseudo-loss' dC/d(s_i) x s_i is passed, resulting in the correct backpropped loss

        :target:
        :irm: ranking evaluation method (method passed directly as an object)
        """

        #modify self.name to include the selected function
        self.name += "_" + irm

        #evaluation methods as lambda functions
        DCG = lambda labels : sum([labels[i-1]/(np.log2(i+1)) for i in range(1,len(labels)+1)])
        nDCG = lambda labels : DCG(labels)/ideal

        ERR = lambda labels : sum([R/r for r,R in enumerate(labels, start=1)]) #see Expected reciprocal rank for graded relevance, equation 5

        #switch between the appropriate
        if irm == "nDCG":
            irm = nDCG
        elif irm == "ERR":
            irm = ERR

        target = target.squeeze() #relevancy labels 0 to 4 for each doc
        loss = torch.zeros(1)
        lambda_i = 0
        delta_i = 0

        #pick documents at random for optimisation (enough documents for 500 comparison operations)
        n = int(np.ceil(np.sqrt(self.random_pairs))) #i.e. sqrt(500) ~= 23
        random_docs = np.random.choice(self.scores.shape[0], n) #i.e. 23*22 ~= 500 random pairs

        #assemble the documents/ scores
        random_docs_labels = [float(target[i]) for i in random_docs] #ordered as per rand selection
        random_docs_ideal_labels = np.sort(random_docs_labels)[::-1] #high to low, perfect order
        ideal = DCG(random_docs_ideal_labels)

        #calculate lambda and delta irm
        for x,i in enumerate(random_docs):
            for y, j in enumerate(random_docs):
                if i == j: continue #skip when i==j

                #swap i,j document relevancy scores, then calculate |delta irm|
                swapped = random_docs_labels.copy()
                swapped[x], swapped[y] = swapped[y], swapped[x]
                # print(irm(swapped, random_docs_ideal_labels))
                delta_ij = irm(swapped) - irm(random_docs_labels)

                #retrieve document scores and calculate lambda_ij
                s_i = self.scores[i]
                s_j = self.scores[j]

                if target[i] > target[j]:
                    S = 1
                elif target[i] == target[j]:
                    S = 0
                else:
                    S = -1

                S = torch.Tensor([S])
                z_i = s_i.detach() ; z_j = s_i.detach() #detach for lambda calc
                lambda_ij = float(self.sigma*(0.5*(1-S) - self.calc_p(z_i, z_j))[0]) #lecture notes equa 34
                # print(lambda_ij)

                delta_i += np.abs(delta_ij)
                lambda_i += lambda_ij

            loss += lambda_i*delta_i*s_i

        return loss

class listwise_cDCG(listWiseModel):
    """
    a child class of listWiseModel, using the cDCG evaluation as part of gradient descent
    """
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 500):
        super().__init__(num_features, scoring_network_layers, dropout, sigma, random_pairs)
        self.name = "LambdaRank_nDCG"

    def loss_function(self,target, irm = "nDCG"):
        super().loss_function(target, irm)

class listwise_ERR(listWiseModel):
    """
    a child class of listWiseModel, using the ERR evaluation as part of gradient descent
    """
    def __init__(self, num_features, scoring_network_layers, dropout = 0.3, sigma = 1, random_pairs = 500):
        super().__init__(num_features, scoring_network_layers, dropout, sigma, random_pairs)
        self.name = "LambdaRank_ERR"

    def loss_function(self,target, irm = "ERR"):
        super().loss_function(target, irm)
