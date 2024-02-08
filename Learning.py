import math
import sys
import time
from random import random

import numpy


class Function:
    def __init__(self, choose):
        self.choose = choose

    def f(self, x):
        match self.choose:
            case 1:
                return 1 / (1 + math.exp(-x))
            case 2:
                if x < 0:
                    return x / 100
                if x > 1:
                    return x / 100
                return x
            case 3:
                if x < 0:
                    return (math.exp(x) - math.exp(-x)) / (100 * (math.exp(x) + math.exp(-x)))
                return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
            case _:
                exit("Wrong func number")

    def f_for_vector_arrays(self, array):
        new_array = [0] * len(array)
        for i in range(len(array)):
            new_array[i] = Function.f(self, array[i])
        return new_array

    def df(self, x):
        match self.choose:
            case 1:
                return self.f(func, x) * (1 - self.f(func, x))
            case 2:
                if x < 0:
                    return 1 / 100
                if x > 1:
                    return 1 / 100
                return 1
            case 3:
                if x < 0:
                    return 0.01 * (1 - self.f(x) ** 2)
                return 1 - self.f(x) ** 2


class Matrix:
    def __init__(self, rows, cols, matrix):
        self.rows = rows
        self.cols = cols
        self.matrix = matrix

    def multiplication_with_vector(self, rows, array):
        if rows != self.cols:
            exit("rows != cols in multiplication")
        result = [0] * self.rows
        for i in range(self.rows):
            for j in range(self.cols):
                result[i] += array[j] * self.matrix[i][j]
        return result

    def sum_vector_arrays(self, a, b):
        if len(a) != len(b):
            exit('err in sum vector arrays')
        c = []
        for i in range(len(b)):
            c.append(a[i] + b[i])
        return c


class Network:
    def __init__(self, size, inputs, func):
        self.network = []
        self.backup_network = []
        self.network.append(inputs)
        self.backup_network.append(inputs)
        self.func = func
        self.size = size

    def __init_weights__(self):
        weights_file = open("weights" + str(self.func.choose) + ".txt")
        weights_file_string_array = weights_file.readlines()
        weights_file_array = []
        for i in range(len(weights_file_string_array)):
            weights_file_array.append(list(map(float, weights_file_string_array[i].split())))
        weights = []
        bios = []
        current_index = 0  # That's because final matrix is 3d, but file is 2d, so I need to convert 3d to 2d by
        # adding special index
        for i in range(len(size) - 1):
            bios.append(numpy.zeros((size[i + 1])))
            weights.append(numpy.zeros((size[i + 1], size[i])))
            for j in range(size[i + 1]):
                bios[i][j] = weights_file_array[current_index + j][0]
                for k in range(size[i]):
                    weights[i][j][k] = weights_file_array[current_index + j][k + 1]
            current_index += size[i + 1]
        self.weights = weights
        self.bios = bios
        weights_file.close()

    def forward_feed(self, stop_index=0):
        for i in range(len(size) - 1 - stop_index):
            a = Matrix
            Matrix.__init__(a, size[i + 1], size[i], self.weights[i])
            b = self.network[i]
            buff = Matrix.sum_vector_arrays(Matrix, Matrix.multiplication_with_vector(a, len(b), b), self.bios[i])
            self.backup_network.append(buff)
            self.network.append(
                Function.f_for_vector_arrays(self.func, buff))

    def save_weights(self, act_random=False):
        weights_file = open('weights' + str(self.func.choose) + '.txt', 'w')
        if not act_random:
            for i in range(len(size) - 1):
                for j in range(size[i + 1]):
                    s = str(self.bios[i][j])
                    for k in range(size[i]):
                        s += ' ' + str(self.weights[i][j][k])
                    s += '\n'
                    weights_file.write(s)
        if act_random:
            print(self.size)
            for i in range(len(size) - 1):
                for j in range(size[i + 1]):
                    s = str(random()/random())
                    for k in range(size[i]):
                        s += ' ' + str(random()/random())
                    s += '\n'
                    weights_file.write(s)
        weights_file.close()

    def back_propagation(self, true_data):
        alpha = 0.9
        dE_by_dS_array = []
        for i in range(len(self.size)-1):
            dE_by_dS_array.append([0]*size[i+1])
        for i in range(len(self.size) - 2, -1, -1):
            for k in range(size[i]):
                for j in range(size[i + 1]):
                    if i==len(self.size) - 2:
                        value =  (self.network[i + 1][j] - true_data[j]) * self.func.df(func, self.backup_network[i + 1][j])
                        self.weights[i][j][k] -= alpha * self.network[i][k] * value
                        self.bios[i][j] -= alpha * value
                        dE_by_dS_array[i][j] = value
                    else:
                        value = 0
                        for jj in range(size[i+1]):
                            value += dE_by_dS_array[i][jj] * self.weights[i][jj][k]
                        value *= self.func.df(func, self.backup_network[i + 1][j])
                        try:
                            self.weights[i][j][k] -= alpha * self.network[i][k] * value
                        except:

                            print(i,j,self.network)
                            w = input()
                        self.bios[i][j] -= alpha * value
                        dE_by_dS_array[i][j] = value


cfg = sys.argv[1:]
inputs = [1, 0]
func = Function
Function.__init__(func, int(cfg[-1]))

num_of_laps = 1000
# or how much time do you want to spend?
time_to_spend = 100  # sec
time_per_lap = 4.5 # sec
time_to_spend_act = False # Activation number of laps by required time
if time_to_spend_act:
    num_of_laps = int(time_to_spend/time_per_lap)
num_of_learning_examples = 4
num_of_test_examples = 0
restart_weights = True
size = list(map(int, cfg[:-1]))
Net = Network
learn_file = open('learn.txt')
array_of_learn_file = learn_file.readlines()

start_time = time.time()
start_lap_time = time.time()
for j in range(num_of_laps):
    num_of_success = 0
    average_error = 0
    num_of_real_success = 0  # difference between real and usual: usual means that you're right with ceil
    # for example 0 or 0 = 0.4 = 0 - success, but not a real success
    for i in range(num_of_learning_examples):
        true_result = list(map(int, array_of_learn_file[2 * i].split()))
        inputs = list(map(int, array_of_learn_file[2 * i + 1].split()))
        Network.__init__(Net, size, inputs, func)
        if restart_weights:
            Network.save_weights(Network, restart_weights)
            restart_weights = False
        Network.__init_weights__(Net)
        Network.forward_feed(Net)
        if num_of_laps < 20:
            print(inputs, Net.network[-1])
        error = 0
        for k in range(len(true_result)):
            error += abs(Net.network[-1][k] - true_result[k])
        average_error += error
        if abs(error) < 0.5:
            num_of_success += 1
        if abs(error) > 0.005:
            Net.back_propagation(Net, true_result)
            Net.save_weights(Net)
        else:
            num_of_real_success += 1
    try:
        if j % (num_of_laps // 20) == 0:
            print(
                f'Epoch {j}: avg err: {average_error / num_of_learning_examples}; avg real success: {num_of_real_success / num_of_learning_examples}; avg success: {num_of_success / num_of_learning_examples}; took {time.time() - start_lap_time} sec')
            start_lap_time = time.time()
    except:
        print(
            f'Epoch {j}: avg err: {average_error / num_of_learning_examples}; avg real success: {num_of_real_success / num_of_learning_examples}; avg success: {num_of_success / num_of_learning_examples}; took {time.time() - start_lap_time} sec')
        start_lap_time = time.time()
    learn_file.close()
print(f'Result: avg err: {average_error / num_of_learning_examples}; avg real success: {num_of_real_success / num_of_learning_examples}; avg success: {num_of_success / num_of_learning_examples}; took {time.time() - start_time} sec; avg time per lap: {(time.time() - start_time)/num_of_laps}')