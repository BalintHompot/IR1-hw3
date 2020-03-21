#A script to pull the evalution scores for val and test data sets, for best model to answer AQ2.2

import pickle
import json

import dataset #provided dataset extraction functions
import evaluate as ev #provided eval functions
from LTRmodels import pointWiseModel

#get the data
data = dataset.get_dataset().get_data_folds()[0]
data.read_data()

#---
#AQ2.2: get score distribution for validation set
#---
with open("./best_models/Pointwise LTR model.model", "rb") as f:
    model = pickle.load(f)

scores = model.score(data.validation)
results_validation = ev.evaluate(data.validation, scores)

#save results
with open("AQ2.2_validation.json", "w") as f:
    json.dump(results_validation, f)

#---
#AQ2.2: get score distribution for test set
#---
with open("./best_models/Pointwise LTR model.model", "rb") as f:
    model = pickle.load(f)

scores = model.score(data.test)
results_validation = ev.evaluate(data.test, scores)

#save results
with open("AQ2.2_test.json", "w") as f:
    json.dump(results_validation, f)

