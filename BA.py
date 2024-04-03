## Imports
import time

import numpy as np
import random
from tqdm import tqdm

import os
import sys

my_lib_path = os.path.abspath("./")
sys.path.append(my_lib_path)

import lstm_module

## Initialize BA parameters
n = 10  # population size
m = 7  # good bees
e = 3  # the elite bees
num = 2  # numbers of the LSTM parameters - number of epochs and number of units
nsp = 2  # number of bees around the good bees
nep = 4  # number of bees around the elite bees
ngh = 1  # neighborhood size
opt = np.zeros([1, num + 1])  # the result (i.e., the optimal bee)
population = np.zeros([n, num + 1])  # list of bees
good = np.zeros([m - e, num + 1])  # list of good bees
elite = np.zeros([e, num + 1])  # list of elite bees

epochs_start = 10
epochs_end = 50
units_start = 8
units_end = 128


## Helper functions
def generate_bee(index):
    # generate the parameters using uniform distribution function
    # first parameter (number of epochs)
    population[index][0] = int(random.uniform(epochs_start, epochs_end))
    # second parameter (number of units)
    population[index][1] = int(random.uniform(units_start, units_end))


def evaluate(bee):
    # train NN and calculate the accuracy
    # model = lstm_module.LSTM(num_epochs=int(bee[0]), num_units=int(bee[1]))
    model = lstm_module.cnn(num_epochs=int(bee[0]), num_units=int(bee[1]))

    accuracy = lstm_module.evaluate_model(model)
    bee[num] = accuracy

    return bee


def sort_bees(unsorted):
    # sort the list of bees descending based on the accuracy column
    return np.array(sorted(unsorted, key=lambda x: x[num], reverse=True))


def local_search(bee, st, amount):
    new_bees = np.zeros([amount, num + 1])
    for i in range(amount):
        # generate new bee in neighborhood of current bee
        nbee = [0, 0, 0]
        # first parameter (number of epochs)
        nbee[0] = int(random.uniform(bee[0] - ngh, bee[0] + ngh))
        # second parameter (number of units)
        nbee[1] = int(random.uniform(bee[1] - ngh, bee[1] - ngh))
        # evaluate new bee
        nbee = evaluate(nbee)
        # if the evaluation of new bee is better than original, keep it else save the original
        if nbee[num] > bee[num]:
            new_bees[i] = nbee
        else:
            new_bees[i] = bee

    new_bees = sort_bees(new_bees)
    return new_bees[0]


start_time = time.time()

## Main
for i in tqdm(range(n)):
    generate_bee(i)
    # evaluate the population
    population[i] = evaluate(population[i])

# sort the population decesntly based on the evaluation value (i.e., accuracy)
population = sort_bees(population)
print(population)

# opt=first bee in the population
opt = population[0]
print("First optimal bee: ", opt)

maxIter = 4
for iter in tqdm(range(maxIter)):
    print("Iteration==========================", iter)
    print(population)

    for i in range(e):
        # call local search for elite and update the population
        population[i] = local_search(population[i], "elite", nep)

    for j in range(i, m):
        # call local search for the remaining good bees and update the population
        population[j] = local_search(population[j], "good", nsp)

    for k in range(j, n):
        # generate the remaining of the population randomly and evaluate
        generate_bee(k)
        # evaluate the population
        population[k] = evaluate(population[k])

    # sort the population
    population = sort_bees(population)
    print(population)
    # opt= the first bee in the population
    opt = population[0]
    print("Final optimal bee: ", opt)

end_time = time.time()
print("Total time for BA: ", end_time - start_time)
