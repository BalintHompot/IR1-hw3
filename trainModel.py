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

################### default params

default_params = {
    "num_features" : data.num_features,
    "epochs":100,
    "layer_num" : 3,
    "layer_size" : 10,
    "lr" : 0.0001,
    "sigma" : 1,
    "random_pairs" : 1000
}

################### parameter ranges for sweep
param_ranges = {
    "learning rates":[0.001, 0.0001, 0.000001],
    "epoch nums":[50,100,200],
    "layer nums":[1,2,3,4],
    "layer sizes":[5,10,50,100],
    "sigmas":[0.5,1,2],
    "random pairs":[500,1000,5000]
}
######################################################

print("--------- Fitting models and testing on set-aside data ------------")
for model in [listwise_ERR]:#[pointWiseModel, RankNetDefualt, RankNetFast, listwise_cDCG, listwise_ERR ]:
    # searching for best params
    best_params_for_model = paramSweep(model, data, default_params, param_ranges)
    # training model with best params
    best_model = construct_and_train_model_with_config(model, data, best_params_for_model)
    # testing the best model
    best_model_results = testModel(best_model, data)
    # saving best model and results
    #TODO


