from cross_validation import crossvalidation_split
import math
import numpy as np
from random import random
from math import *
import GA as ga
import dataset
import ANN

# save activation and derivative
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some prediction


class MLP(object):

    def __init__(self, num_inputs=30, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initiate random weight
        weight = []
        for i in range(len(layers)-1):
            w = np.random.uniform(low=-0.1, high=0.1, size=(hidden_layers))
            weight.append(w)
        self.weight = weight


    def GeneticAlgorithm(self, population, numMating, num_generations, mutation_percent):
        for generation in range(num_generations):
            print("Generation : ", generation)

            # converting the solutions from being vectors to matrices.
            pop_weights_mat = ga.vector_to_mat(population, 
            pop_weights_mat)

            # Measuring the fitness of each chromosome in the population.
            fitness = ANN.fitness(pop_weights_mat, 
            dataset.Int_Input, 
            dataset.Output_data, 
            activation="sigmoid")
            accuracies[generation] = fitness[0]
            print("Fitness")
            print(fitness)

            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(population, 
                                            fitness.copy(), 
                                            num_generations)
            print("Parents")
            print(parents)

            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(parents,
            offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))
            print("Crossover")
            print(offspring_crossover)

            # Adding some variations to the offsrping using mutation.
            offspring_mutation = ga.mutation(offspring_crossover, 
            mutation_percent=mutation_percent)
            print("Mutation")
            print(offspring_mutation)

            # Creating the new population based on the parents and offspring.
            pop_weights_vector[0:parents.shape[0], :] = parents
            pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

        pop_weights_mat = ga.vector_to_mat(dataset.Int_Input, pop_weights_mat)
        best_weights = pop_weights_mat [0, :]
        acc, predictions = ANN.predict_outputs(best_weights, dataset.Int_Input, dataset.Output_data, activation="sigmoid")
        print("Accuracy of the best solution is : ", acc)
        

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for j, input in enumerate(inputs):
                target = targets[j]

                # forward prop
                output = self.GeneticAlgorithm(input)

                # calculate error
                error = target - output

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i+1))

    print("Training complete!")
    print("=====")

    def gradient_descent(self, learning_rate=1):
        for i in range(len(self.weight)):
            weights = self.weight[i]
            derivatives = self.derivative[i]
            weights += derivatives * learning_rate

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def relu(inpt):
    result = inpt
    result[inpt<0] = 0
    return result



if __name__ == "__main__":
    Input = []
    Output = []
    error_log = []
    k=10
    folds = crossvalidation_split(dataset, k)
    for i in range(k):
        _k = len(folds[i])
        # create a dataset to train a network for the sum operation
        # print(folds[0][0])
        items = np.array([folds[i][j][0:len(folds[i][j])-1] for j in range(len(folds[i]))])
        targets = np.array([folds[i][j][len(folds[i][j])-1] for j in range(len(folds[i]))])
        # print(items[0][0])
        # print(targets[0])

        # create a Multilayer Perceptron with one hidden layer
        mlp = MLP(30, [8,8], 2)

        # train network
        mlp.train(items, targets, 1000, 0.8)

        _input = folds[i][_k-1][0:len(folds[i][_k-1])-1]
        input = np.array(_input)
        target = np.array((folds[i][_k-1][len(folds[i][_k-1])-1]))

        output = mlp.GeneticAlgorithm(input)
        Input.append(dataset.denomallize(target))
        Output.append(dataset.denomallize(output))

        print()
        print("if 2 station  have water level = {}  \nIn the next 7 hours the water level should be {}".format(dataset.denomallize(input[0:8]), dataset.denomallize(output)))
        print("but actually should be {}".format(dataset.denomallize(target)))
        print()

        error_log.append(abs(dataset.denomallize(target)-dataset.denomallize(output))*100/dataset.denomallize(target))


    for i in range(10):
        print("error round {} : {:.2f}%".format(i+1,error_log[i][0]))