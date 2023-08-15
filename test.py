from river.tree import HoeffdingTreeClassifier
from river import stream
from river import metrics
from river import evaluate

import time
import datetime as dt
import pandas as pd
import os
import random

import argparse
import matplotlib.pyplot as plt


PATH_TO_DATA = "C:/Users/Bayde/Desktop/Hoeffding_Tree_Project_Data/datasets_csv"


def model_size(model):
    if str(model) == 'HoeffdingTreeClassifier':
        return model.n_nodes
    # In case we are interested in counting the number of nodes from all trees.
    elif str(model) == 'ARFClassifier':
        nodes = 0
        for m in model.models:
            nodes += m.model.n_nodes
        return nodes
    return -1


def run_experiment(tree, dataset):
    accuracy = metrics.Accuracy()
    start = time.perf_counter()

    X, Y = dataset[:, :-1], dataset[:, -1]
    # Downsampling the data
    # Select every 20th element starting from the first element
    # Useful during development, when you are making sure your
    #    algorithm works before doing the actual experiments.

    X, Y = X[::100], Y[::100]  # TODO: <--- Remember to remove downsampling

    accuracies = []
    predictions = []
    plotacc = []

    for x in range(len(ensemble)):
        accuracies.append(metrics.Accuracy())
        predictions.append([])

    en_acc = metrics.Accuracy()
    en_predict = []

    candidate = new_tree()
    cand_acc = metrics.Accuracy()
    cand_pre = []

    ds = stream.iter_array(X, Y)
    counter = 0
    for (x, y) in ds:

        #trees:
        for i in range(len(ensemble)):
            yp = ensemble[i].predict_one(x)
            predictions[i].append(yp)
            ensemble[i].learn_one(x, y)
            accuracies[i].update(y, yp)

        #candidate tree:
        cyp = candidate.predict_one(x)
        cand_pre.append(cyp)
        candidate.learn_one(x, y)
        cand_acc.update(y, cyp)

        #ensemble:
        en_predict.append(vote(predictions, counter, accuracies))
        en_acc.update(y, en_predict[counter])
        plotacc.append(en_acc.get())

        if counter % window_size == 0:  # Candidate tree time

            # if tree better than weakest swap in
            if cand_acc.get() < min(acc.get() for acc in accuracies):
                #print("IGNORED")
                continue
            else:
                lowest_index = 99
                lowest_val = 1
                for x in range(len(ensemble)):
                    if accuracies[x].get() < lowest_val:
                        lowest_index = x
                        lowest_val = accuracies[x].get()
                ensemble[lowest_index] = candidate
                #print("SWAPPED (", cand_acc.get(), " > ", lowest_val, ")")

            candidate = new_tree()  # instantiate new candidate

        counter = counter + 1
        #accuracy.update(y, yp)
        #print("Ensemble: {:.2f} Candidate: {:.2f} Trees: {}".format(en_acc.get(), cand_acc.get(), accuracies))

    return (en_acc.get(), plotacc)
    #return accuracy.get(), dt.timedelta(seconds=time.perf_counter() - start).seconds, model_size(model)


def vote(pred, count, acc):
    w = {}  # votes
    for model_num in range(len(pred)):
        # Get the prediction for the current model and instance
        prediction = pred[model_num][count]
        accuracy = acc[model_num].get()

        if prediction not in w:
            w[prediction] = 0
        w[prediction] += accuracy

    # Grab max weighted value class
    ensemble_prediction = max(w, key=w.get)
    return ensemble_prediction


def new_tree():
    # Random grace period from the range (10, 200) with step = 10
    gp = random.choice(range(10, 201, 10))

    # Random split_confidence from the range (0.0, 1.0) with step = 0.05
    sc = random.choice([i * 0.05 for i in range(1, 21)])

    # Random tau from the range (0.0, 1.0) with step = 0.05
    t = random.choice([i * 0.05 for i in range(21)])

    new_model = HoeffdingTreeClassifier(
        grace_period=gp,
        delta=sc,
        tau=t)
    return new_model


def ensem(num_learners):
    ensemble = []
    for x in range(num_learners):
        tree = new_tree()
        ensemble.append(tree)
    return ensemble


def main(num_learners, window_size, seed):
    random.seed(seed)
    # Optional HyperParameter Arguments for base HT Classifier, default is (10,0,0)
    print(f"Number of learners: {num_learners}")
    print(f"Candidate window size: {window_size}")

    datasets = ['electricity.csv', 'SEA_abrupt.csv', 'SEA_gradual.csv',
                'RTG_2abrupt.csv', 'covtype.csv']

    dataset_res = []

    dataset_res.append(datasets)
    text = ("[ " + "Learners: " + str(num_learners) + "Window: " +
            str(window_size) + "Seed: " + str(seed) + " ]")
    dataset_res.append(text)

    for name in datasets:  # For Each Dataset
        tree = HoeffdingTreeClassifier(
            grace_period=50,
            delta=0.5,
            tau=0.5)
        dataset = pd.read_csv(os.path.join(PATH_TO_DATA, name)).to_numpy()
        results = run_experiment(tree, dataset)
        dataset_res.append(results)
    print("---DONE---")
    return(dataset_res)


if __name__ == '__main__':
    main()
