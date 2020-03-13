import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
from LTRmodels import *


data = dataset.get_dataset().get_data_folds()[0]
data.read_data()

print('Number of features: %d' % data.num_features)
print('Number of queries in training set: %d' % data.train.num_queries())
print('Number of documents in training set: %d' % data.train.num_docs())
print('Number of queries in validation set: %d' % data.validation.num_queries())
print('Number of documents in validation set: %d' % data.validation.num_docs())
print('Number of queries in test set: %d' % data.test.num_queries())
print('Number of documents in test set: %d' % data.test.num_docs())

# initialize a random model
random_model = pointWiseModel(data.num_features, [10,10,10])

all_scores = random_model.score(data.train)

# rank every query for all scores (1d vector ordered by query ordering in dataset)
all_rankings, all_inverted_rankings = random_model.rank(all_scores, data.train)

qid = 1
s_i, e_i = data.train.query_range(qid)

# to rank only a single query use rank_and_invert
query_ranking, query_inverted_ranking = rnk.rank_and_invert(all_scores[s_i:e_i])

assert np.all(np.equal(query_ranking, all_rankings[s_i:e_i]))
assert np.all(np.equal(query_inverted_ranking, all_inverted_rankings[s_i:e_i]))

print('-------')
print('Looking at query with id: %d' % qid)
print('Number of documents in query %d: %d' % (qid, data.train.query_size(qid)))
print('Scores for query %d: %s' % (qid, all_scores[s_i:e_i]))
print('Ranking for query %d: %s' % (qid, all_rankings[s_i:e_i]))
print('Inverted ranking for query %d: %s' % (qid, all_inverted_rankings[s_i:e_i]))

validation_scores = random_model.score(data.validation)
print('------')
print('Evaluation on entire validation partition.')
results = evl.evaluate(data.validation, validation_scores, print_results=True)
