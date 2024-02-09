import subprocess
import math
import sys
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
            case 4:
                print('FTS func in process')
            case _:
                exit("Wrong func number")

    def f_for_vector_arrays(self, array):
        new_array = [0] * len(array)
        for i in range(len(array)):
            new_array[i] = Function.f(self, array[i])
        return new_array


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
    def __init__(self, size, channels, operations, kernel_size, pad_n_pool_func, act_func_choices):
        self.size = size
        self.channels = channels
        self.operations = operations
        self.kernel_size = kernel_size
        self.pad_n_pool_func = pad_n_pool_func
        self.act_func_choices = act_func_choices
        array_1 = [Function]*len(act_func_choices)
        for i in range(len(act_func_choices)):
            array_1[i].__init__(array_1[i], act_func_choices[i])
        self.act_func_array = array_1
        self.__init_weights__(self)

    def __init_weights__(self):
        self.save_weights(self)
        w_file = open('weights.txt')
        weights_file = w_file.readlines()
        self.MP_weights = []
        self.kernel_weights = []
        w_file.close()
        for i in range(len(weights_file)):
            s = weights_file[i]
            array_2 = []
            if self.operations[i] == 0:
                s.split(';')

    def save_weights(self, act_random = True):
        w_file = open('weights.txt', 'w')
        if act_random:
            counter = 0
            for i in range(len(self.size) -1):
                if self.operations[i] == 0:
                    for ii in range(self.channels[i]):
                        for j in range(self.kernel_size[i]):
                            s = ''
                            for k in range(self.kernel_size[i]):
                                s += str(random()) + ' '
                                counter += 1
                            w_file.write(s[:-1] + ';')
                        w_file.write(';')
                elif self.operations[i] == 2:
                    for j in range(self.size[i]):
                        s = ''
                        for k in range(self.size[i+1]):
                            counter += 1
                            s += str(random()) + ' '
                        w_file.write(s[:-1] + ';')
                    w_file.write('\n')
                w_file.write('\n')
            print(counter)
        w_file.close()

    def forward_feed(self):
        return 0


"""
shortnames:
MP - multilayer perceptron
config.txt structure:
0 line: number of neurons in row/col on every layer, then num of neurons on every layer of MP
1 line: num of channels on every layer, except MP 
2 line: operation on layer (0 - convolution; 1 - pooling; 2 - MP)
3 line: length / width of kernel ( 2x2 type only 2), except MP
4 line: for convolution one side of padding( 0 - no pad); for pooling 0 - max; 1 - avg
5 line: activate function (1 - sigmoid; 2 - modRELu; 3 - th; 4 - FLT) for convolution and MP; for padding - stride num
6 line: using mode (0 - if you want to use; 1 - if you want to learn)

"""
file = open("config.txt")
readfile = file.readlines()
file.close()

if readfile[-1] == '1':
    subprocess.run(['python', 'Learning.py'])

main_size = list(map(int, readfile[0].split()))
main_channels = list(map(int, readfile[1].split()))
main_operations = list(map(int, readfile[2].split()))
main_kernel_size = list(map(int, readfile[3].split()))
main_pad_n_pool_func = list(map(int, readfile[4].split()))
main_act_func_choices = list(map(int, readfile[5].split()))
Net = Network
Net.__init__(Net, main_size, main_channels, main_operations, main_kernel_size, main_pad_n_pool_func, main_act_func_choices)
