import subprocess
import math
import sys
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
        if len(a)!=len(b):
            exit('err in sum vector arrays')
        c=[]
        for i in range(len(b)):
            c.append(a[i]+b[i])
        return c


class Network:
    def __init__(self):
        self.f = 1

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

size = list(map(int, readfile[0].split()))
channels = list(map(int, readfile[1].split()))
operations = list(map(int, readfile[2].split()))



