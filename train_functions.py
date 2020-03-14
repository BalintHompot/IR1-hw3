
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

def testModel(model, data, epochs, optimizer):
    test_scores = model.score(data.test)
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    print('Final performance of ' + model.name + ' on test partition.')
    results = evl.evaluate(data.test, test_scores, print_results=True)
    return model, results["ndcg"]


def paramSweep(model, data):
    best_config = {}
    best_ndcg = -10000
    epochs = 200
    learning_rates = [0.01,0.001,0.0001,0.00001]
    epoch_nums = [50,100,200,500]
    layer_nums = [1,2,3,4]
    layer_neurons = [5,10,50,100]

    ### to save time we optimize for the params separately (which might not be optimal, but we don't have to run hundreds of models)
    for lr in learning_rates:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
        if(current_ndcg > best_ndcg):
            best_config["learning rate"] = lr

    optimizer.lr = 0.0001
    best_ndcg = -1000
    for epochs in epoch_nums:
        current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
        if(current_ndcg > best_ndcg):
            best_config["epochs"] = lr

    epochs = 200
    layer_neuron = 10
    best_ndcg = -1000
    for layer_num in layer_nums:
        hidden_layers = [layer_neuron for i in range(layer_num)]
        model.constructScoringNetwork(hidden_layers)
        current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
        if(current_ndcg > best_ndcg):
            best_config["number of layers"] = layer_num

    layer_num = 3
    best_ndcg = -1000
    for layer_neuron in layer_neurons:
        hidden_layers = [layer_neuron for i in range(layer_num)]
        model.constructScoringNetwork(hidden_layers)
        current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
        if(current_ndcg > best_ndcg):
            best_config["number of neurons per layer"] = layer_neuron

    

    ## additional params if not using pointwise
    if model.name != "Pointwise LTR model":
        sigmas = [0.5,1,2]
        pair_batch_sizes = [100,1000,10000]

        layer_neuron = 10
        hidden_layers = [layer_neuron for i in range(layer_num)]
        best_ndcg = -1000
        model.constructScoringNetwork(hidden_layers)

        for sigma in sigmas:
            model.sigma = sigma
            current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
            if(current_ndcg > best_ndcg):
                best_config["sigma"] = sigma

        for pair_batch_size in pair_batch_sizes:
            model.random_pairs = pair_batch_size
            current_model, current_ndcg = trainModel(model, data, epochs, optimizer)
            if(current_ndcg > best_ndcg):
                best_config["number of random pairs"] = pair_batch_size

    return best_config

