import numpy
import random


class Node:

    data_list = []
    weights = []
    bias = 0

    def __init__(self, data_list):
        self.data_list = data_list
        self.weights, self.bias = self.__randomize_weights(data_list)

    # activation function for this neuron: sigmoid
    def __sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    # returns randomize weights
    def __randomize_weights(self, data):
        weights = []
        for x in range(len(data[0]) - 1):
            weights.append(numpy.random.randn())

        bias = numpy.random.randn()
        return weights, bias

    # runs the node
    def run_node(self):
        _sum = self.bias
        for weight in self.weights:
            current_data = float(input("enter data: "))
            _sum += weight * current_data
        return self.__sigmoid(_sum)

    # resets the current data
    def reset_data(self, new_data):
        self.data_list = new_data

    # this function trains the neuron
    def train(self, lrn_rate, iterations):
        for i in range(iterations):

            # define what the outcome of the node should be in current iteration
            current_d = self.data_list[random.randint(0, len(self.data_list) - 1)]
            s = self.bias

            # sums up all weights*data
            counter = 0
            for w in self.weights:
                s += w * current_d[counter]
                counter += 1

            # gets the result for current data
            result = current_d[-1]
            # gets prediction
            prediction = self.__sigmoid(s)

            # define derivatives of cost and sigmoid functions
            dcost = 2 * (prediction - result)
            dsigmoid = prediction * (1 - prediction)

            # derive the cost function by every weight
            for j in range(len(self.weights)):
                # defines the derivative of weight*data
                dw = current_d[j]
                # defines the derivative of cost(current_weight)
                dcost_by_w = dcost * dsigmoid * dw
                # defines vector and improves weight
                self.weights[j] = self.weights[j] - lrn_rate * dcost_by_w

            # does the same for the bias
            dbias = 1
            dcost_by_bias = dcost * dsigmoid * dbias
            self.bias = self.bias - lrn_rate * dcost_by_bias