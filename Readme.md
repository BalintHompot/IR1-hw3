# Learning to Rank
This repository contains code for comparing different approaches to offline learning to rank. This is the code for a course assignment for Information Retrieval 1 at the University of Amsterdam, see detailed description in ir1_2020_hw3.pdf.

In the project we compare three approaches towards LTR:

* pointwise approach: predicting relevancy without focusing on ranking directly
* pairwise approach: optimizing for correct pairwise comparison. The project includes a naive and a sped-up implementation of the pairwise model.
* listwise approach: we don't only focus on pairwise comarison, but also on how swapping a pair would affect the evaluation of the list as a whole.

## Structure
### trainModel.py
is the main function of the project. It contains the definition of default parameters and parameter ranges to be searched and the list of models that we want to test. Then, for all models it creates an instance, runs hyperparameter search and stores the best model and best configuration in best_models and best_configs folders, then tests the models and stores the results in the best_results folder. 

### train_functions.py
is the file that contains the code for all functions that are called in the trainModel script.

### LTRModels.py
contains the class definition for all LTR model (Pointwise, RankNet, RankNetFast, Listwise) that are instantiated in the trainModel script.

### evaluate.py
contains script for IR evaluation metrics such as precision, recall, dcg and ndcg.