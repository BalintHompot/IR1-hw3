#A script to plot evaluate.py mean, sd results 

import json
import matplotlib.pyplot as plt
import numpy as np

#pull the evaluate.py results from jsons
with open("AQ2.2_validation.json", "r") as f:
    validation_results = json.load(f)

with open("AQ2.2_test.json", "r") as f:
    test_results = json.load(f)


#plot in 5 plots of the following
set1 = ['dcg', 'dcg@03', 'dcg@05', 'dcg@10', 'dcg@20']
set2 = ['ndcg', 'ndcg@03', 'ndcg@05', 'ndcg@10', 'ndcg@20']
set3 = ['precision@01', 'precision@03', 'precision@05', 'precision@10', 'precision@20']
set4 = ['recall@01', 'recall@03', 'recall@05', 'recall@10', 'recall@20']
set5 = [ 'relevant rank', 'relevant rank per query']

for s in [set1, set2, set3, set4, set5]:

    for stat in ["mean", "sd"]:

        if stat == "mean":
            index = 0
        elif stat == "sd":
            index = 1


        data = [[], []]
        labels = [[],[]]
        for k in s:

            #assemble data
            data[0].append(test_results[k][index])
            data[1].append(validation_results[k][index])

            #assemble labels
            labels[0].append(k)
            labels[1].append(k)

        #create plot
        n_groups =len(data[0]) 

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.20
        opacity = 0.8

        test = plt.bar(index, data[0], bar_width,
        alpha=opacity,
        color='b',
        label='test')

        val = plt.bar(index + bar_width, data[1], bar_width,
        alpha=opacity,
        color='g',
        label='validation')

        plt.xlabel('score')
        plt.ylabel('mean')
        plt.title('comparison of test and validation scores')
        plt.xticks(index + bar_width, labels[0])
        plt.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"AQ2.2_barchart_"+ stat + f"_{s[0]}.png")
