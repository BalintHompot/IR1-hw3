import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
from tqdm import tqdm
from LTRmodels import *
from train_functions import *
DEVICE = "cpu"


data = dataset.get_dataset().get_data_folds()[0]
data.read_data()
epochs = 100

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

pointwise_model = pointWiseModel(data.num_features, [10,10,10])
ranknet_default = RankNetDefualt(data.num_features, [10,10,10])

print("--------- Fitting models and testing on set-aside data ------------")
for model in [pointwise_model, ranknet_default]:
    # searching for best params
    best_params_for_model = paramSweep(model, data)
    # training model with best params
    best_model = construct_and_train_model_with_config(model, data, best_params_for_model)
    # testing the best model
    best_model_results = testModel(best_model, data)
    # saving best model and results
    #TODO


