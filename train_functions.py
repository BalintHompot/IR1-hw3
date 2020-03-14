import ranking as rnk
import evaluate as evl
import numpy as np
import torch
from tqdm import tqdm
import json
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

    ### testing on test set and returning ndcg for it
    validation_scores = model.score(data.validation)
    print('------')
    print('Evaluation of ' + model.name + ' on validation partition.')
    results = evl.evaluate(data.validation, validation_scores, print_results=False)
    print(results["ndcg"])
    return model, results["ndcg"]

def testModel(model, data):
    test_scores = model.score(data.test)
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    print('Final performance of ' + model.name + ' on test partition.')
    results = evl.evaluate(data.test, test_scores, print_results=True)
    return results

def construct_and_train_model_with_config(modelClass, data, config):
    epochs = config["epochs"]
    hidden_layers = [config["number of neurons per layer"] for i in range(config["number of layers"])]
    num_features = config["num features"]
    if model.name != "Pointwise LTR model":
        sigma = config["sigms"]
        random_pairs = config["number of random pairs"]
    else:
        sigma = 1
        random_pairs = 500
    
    model = modelClass(num_features, hidden_layers, sigma=sigma, random_pairs=random_pairs)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning rate"])

    trained_model, train_results = trainModel(model, data, epochs, optimizer)
    return trained_model

def paramSweep(modelClass, data, default_config, param_ranges):
    best_config = {"num features":data.num_features}
    best_ndcg = -10000

    learning_rates = param_ranges["learning rates"]
    epoch_nums = param_ranges["epoch nums"]
    layer_nums = param_ranges["layer nums"]
    layer_neurons = param_ranges["layer sizes"]
    ## additional for pairwise and listwise
    sigmas = param_ranges["sigmas"]
    pair_batch_sizes = param_ranges["random pairs"]


    ### to save time we optimize for the params separately (which might not be optimal, but we don't have to run hundreds of models)
    print("............................")
    print("optimizing learning rate")
    print("............................")
    for lr in learning_rates:
        hidden_layers = [default_config["layer_size"] for i in range(default_config["layer_num"])]
        model = modelClass(default_config["num_features"], hidden_layers, sigma=default_config["sigma"], random_pairs=default_config["random_pairs"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        current_model, current_ndcg = trainModel(model, data, default_config["epochs"], optimizer)
        if(current_ndcg[0] > best_ndcg):
            best_config["learning rate"] = lr
    print("............................")
    print("finished sweeping for learning rate, best value: " + str(best_config["learning rate"]))
    print("............................")

    best_ndcg = -1000
    print("optimizing epoch num")
    print("............................")
    for epochs in epoch_nums:
        hidden_layers = [default_config["layer_size"] for i in range(default_config["layer_num"])]
        model = modelClass(default_config["num_features"], hidden_layers, sigma=default_config["sigma"], random_pairs=default_config["random_pairs"])
        optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
        current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
        if(current_ndcg[0] > best_ndcg):
            best_config["epochs"] = epochs
    print("............................")
    print("finished sweeping for epoch num, best value: " + str(best_config["epochs"]))
    print("............................") 

    best_ndcg = -1000

    print("optimizing layer num")
    print("............................")
    for layer_num in layer_nums:
        hidden_layers = [default_config["layer_size"] for i in range(layer_num)]
        model = modelClass(default_config["num_features"], hidden_layers, sigma=default_config["sigma"], random_pairs=default_config["random_pairs"])
        optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
        current_model, current_ndcg = trainModel(model, data, default_config["epochs"], optimizer)
        if(current_ndcg[0] > best_ndcg):
            best_config["number of layers"] = layer_num
    print("............................")
    print("finished sweeping for number of layers, best value: " + str(best_config["number of layers"]))
    print("............................")

    layer_num = 3
    best_ndcg = -1000
    print("optimizing layer size")
    print("............................")
    for layer_neuron in layer_neurons:
        hidden_layers = [layer_neuron for i in range(default_config["layer_num"])]
        model = modelClass(default_config["num_features"], hidden_layers, sigma=default_config["sigma"], random_pairs=default_config["random_pairs"])
        optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
        current_model, current_ndcg = trainModel(model, data, default_config["epochs"], optimizer)
        if(current_ndcg[0] > best_ndcg):
            best_config["number of neurons per layer"] = layer_neuron
    print("............................")
    print("finished sweeping for neurons per layer, best value: " + str(best_config["number of neurons per layer"]))
    print("............................")
    

    ## additional params if not using pointwise
    if model.name != "Pointwise LTR model":
        layer_neuron = 10
        hidden_layers = [layer_neuron for i in range(layer_num)]
        best_ndcg = -1000
        model.constructScoringNetwork(hidden_layers)

        print("optimizing sigma")
        print("............................")
        for sigma in sigmas:
            hidden_layers = [default_config["layer_size"] for i in range(default_config["layer_num"])]
            model = modelClass(default_config["num_features"], hidden_layers, sigma=sigma, random_pairs=default_config["random_pairs"])
            optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])            
            current_model, current_ndcg = trainModel(model, data, default_config["epochs"], optimizer)
            if(current_ndcg[0] > best_ndcg):
                best_config["sigma"] = sigma
        print("............................")
        print("finished sweeping for sigma, best value: " + str(best_config["sigma"]))
        print("............................")
        best_ndcg = -1000

        print("optimizing number of pairs drawn")
        print("............................")
        for pair_batch_size in pair_batch_sizes:
            hidden_layers = [default_config["layer_size"] for i in range(default_config["layer_num"])]
            model = modelClass(default_config["num_features"], hidden_layers, sigma=default_config["sigma"], random_pairs=pair_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])            
            current_model, current_ndcg = trainModel(model, data, default_config["epochs"], optimizer)
            if(current_ndcg[0] > best_ndcg):
                best_config["number of random pairs"] = pair_batch_size
        print("............................")
        print("finished sweeping for drawn random pair number, best value: " + str(best_config["number of random pairs"]))
        print("............................")

    with open("./best_configs/" + model.name + "_best_config.json", "w+") as writer:
        json.dump(best_config, writer, indent=1)

    return best_config

