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
model = pairWiseModel(data.num_features, [10,10,10])

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

# initialize a random model
trainModel(model, data, epochs, optimizer)


