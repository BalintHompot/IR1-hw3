import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
from tqdm import tqdm
from LTRmodels import *
DEVICE = "cpu"

def trainModel(model, data, epochs, optimizer):
    print("======================== " + model.name + "========================")

    labels = torch.Tensor(data.train.label_vector).to(DEVICE).unsqueeze(1)

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        all_scores = model.score(data.train)
        loss = model.loss_function(labels)
        loss.backward()
        optimizer.step()



data = dataset.get_dataset().get_data_folds()[0]
data.read_data()
epochs = 500
model = pointWiseModel(data.num_features, [10,10,10])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

# initialize a random model
trainModel(model, data, epochs, optimizer)

validation_scores = model.score(data.validation)
print('------')
print('Evaluation on entire validation partition.')
results = evl.evaluate(data.validation, validation_scores, print_results=True)