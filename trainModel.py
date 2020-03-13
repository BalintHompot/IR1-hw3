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
    labels_val = torch.Tensor(data.validation.label_vector).to(DEVICE).unsqueeze(1)

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        all_scores = model.score(data.train)
        loss = model.loss_function(labels)
        loss.backward()
        optimizer.step()
        if epoch%20 == 0:
            print("validation ndcg at epoch " + str(epoch))
            model.eval()
            validation_scores = model.score(data.validation)
            results = evl.evaluate(data.validation, validation_scores, print_results=False)
            print(results["ndcg"])




data = dataset.get_dataset().get_data_folds()[0]
data.read_data()
epochs = 100
model = pairWiseModel(data.num_features, [10,10,10])
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

test_scores = model.score(data.test)
print('------')
print('Evaluation on entire test partition.')
results = evl.evaluate(data.test, test_scores, print_results=True)
