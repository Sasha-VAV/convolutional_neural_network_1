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


class Network:
    def __init__(self, size, channels, operations, kernel_size, pad_n_pool_func, act_func_choices):
        self.size = size
        self.channels = channels
        self.operations = operations
        self.kernel_size = kernel_size
        self.pad_n_pool_func = pad_n_pool_func
        self.act_func_choices = act_func_choices
        array_1 = [Function] * len(act_func_choices)
        for i in range(len(act_func_choices)):
            array_1[i].__init__(array_1[i], act_func_choices[i])
        self.act_func_array = array_1
        self.__init_weights__(self)

    def __init_weights__(self):
        self.save_weights(self, True)
        w_file = open('weights.txt')
        weights_file = w_file.readlines()
        self.MP_weights = []
        self.MP_bios = []
        self.kernel_weights = []
        w_file.close()
        for i in range(len(weights_file)):
            s = weights_file[i]
            array_2 = []
            if self.operations[i] == 0:
                array_3 = list(s.split(';;'))
                for j in range(self.channels[i + 1]):
                    array_4 = array_3[j].split(';')
                    array_5 = []
                    for k in range(self.kernel_size[i]):
                        array_5.append(list(map(float, array_4[k].split())))
                    array_2.append(array_5)
                self.kernel_weights.append(array_2)
            elif self.operations[i] == 2:
                '''if self.channels[i + 1] > 1:
                    exit("more than 1 channel on MP")'''
                self.MP_bios.append(list(map(float, s[s.find(';b')+2:].split())))
                s = s[:s.find(';b')]
                array_1 = list(s.split(';'))
                for j in range(self.size[i]):
                    array_2.append(list(map(float, array_1[i].split())))
                self.MP_weights.append(array_2)
            else:
                self.kernel_weights.append(0)
        self.save_weights(self)

    def save_weights(self, act_random=False):
        w_file = open('weights.txt', 'w')
        if act_random:
            counter = 0
            for i in range(len(self.size) - 1):
                if self.operations[i] == 0:
                    for ii in range(self.channels[i + 1]):  # !!!
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
                        for k in range(self.size[i + 1]):
                            counter += 1
                            s += str(random()) + ' '
                        w_file.write(s[:-1] + ';')
                    w_file.write('b')
                    s = ''
                    for k in range(self.size[i + 1]):
                        counter += 1
                        s += str(random()) + ' '
                    w_file.write(s[:-1])
                w_file.write('\n')
            print(counter)
        else:
            counter = 0
            i_for_MP = 0
            for i in range(len(self.size) - 1):
                if self.operations[i] == 0:
                    for ii in range(self.channels[i + 1]):  # !!!
                        for j in range(self.kernel_size[i]):
                            s = ''
                            for k in range(self.kernel_size[i]):
                                s += str(self.kernel_weights[i][ii][j][k]) + ' '
                                counter += 1
                            w_file.write(s[:-1] + ';')
                        w_file.write(';')
                elif self.operations[i] == 2:
                    for j in range(self.size[i]):
                        s = ''
                        for k in range(self.size[i + 1]):
                            counter += 1
                            s += str(self.MP_weights[i_for_MP][j][k]) + ' '
                        w_file.write(s[:-1] + ';')
                    w_file.write('b')
                    s = ''
                    for k in range(self.size[i + 1]):
                        counter += 1
                        s += str(self.MP_bios[i_for_MP][k]) + ' '
                    w_file.write(s[:-1])
                    i_for_MP += 1
                w_file.write('\n')
            print(counter)
        w_file.close()

    @staticmethod
    def multiplication_value_arr_with_matrix(value_array, array, padding=0):
        new_array = []
        for i in range(padding, len(value_array) - padding):
            array_1 = []
            for j in range(padding, len(value_array[i]) - padding):
                current_sum = 0
                delta = len(array) // 2
                for ii in range(-delta, delta + 1):
                    for jj in range(-delta, delta + 1):
                        if 0 <= i + ii < len(value_array) and 0 <= j + jj < len(value_array):
                            current_sum += value_array[i + ii][j + jj] * array[ii + delta][jj + delta]
                array_1.append(current_sum)
            new_array.append(array_1)
        return new_array

    @staticmethod
    def sum_matrix(array):
        new_array = []
        for i in range(len(array[0])):
            array_1 = []
            for j in range(len(array[0][i])):
                curr_sum = 0
                for k in range(len(array)):
                    curr_sum += array[k][i][j]
                array_1.append(curr_sum)
            new_array.append(array_1)
        return new_array

    def use_kernel(self, current_index, current_array):
        output_array = []
        for i in range(self.channels[current_index + 1]):
            array_1 = []
            kernel_array = self.kernel_weights[current_index][i]
            for j in range(self.channels[current_index]):
                array_1.append(self.multiplication_value_arr_with_matrix(current_array[j], kernel_array,
                                                                         self.pad_n_pool_func[current_index]))
            output_array.append(self.sum_matrix(array_1))
        return output_array

    def use_pooling(self, current_index, current_array):
        new_array = []
        current_size = self.size[current_index]
        current_kernel_size = self.kernel_size[current_index]
        current_scale = current_size // current_kernel_size
        stride = self.act_func_choices[current_index]
        if self.pad_n_pool_func[current_index] == 0:
            for i in range(0, current_size - stride + 1, stride):
                array_1 = []
                for j in range(0, current_size - stride + 1, stride):
                    max_value = current_array[i][j]
                    for ii in range(current_kernel_size):
                        for jj in range(current_kernel_size):
                            max_value = max(
                                current_array[i + ii][j + jj],
                                max_value)
                    array_1.append(max_value)
                new_array.append(array_1)
        elif self.pad_n_pool_func[current_index] == 1:
            for i in range(0, current_size - stride + 1, stride):
                array_1 = []
                for j in range(0, current_size - stride + 1, stride):
                    sum_value = 0
                    for ii in range(current_kernel_size):
                        for jj in range(current_kernel_size):
                            try:
                                sum_value += current_array[i + ii][j + jj]
                            except:
                                print('xx')
                    array_1.append(sum_value / (current_kernel_size ** 2))
                new_array.append(array_1)
        elif self.pad_n_pool_func[current_index] == -1:
            '''USE YOUR FUNCTION'''
        else:
            exit("we don't have this pooling function at this moment")
        return new_array

    def use_mp(self, current_index, current_mp_index, current_array):
        new_array = [0] * self.size[current_index + 1]
        for i in range(self.size[current_index]):
            for j in range(self.size[current_index + 1]):
                new_array[j] += current_array[i] * self.MP_weights[current_mp_index][i][j]
        for j in range(self.size[current_index + 1]):
            new_array[j] += 1 * self.MP_bios[current_mp_index][j]
        return new_array

    def use_flatten(self, current_array):
        new_array = []
        for i in range(len(current_array)):
            for j in range(len(current_array[i])):
                for k in range(len(current_array[i][j])):
                    new_array.append(current_array[i][j][k])
        return new_array

    def forward_feed(self):
        f_input = open('input.txt')
        array_1 = f_input.readlines()
        current_array = []
        i_for_mp = 0
        for k in range(self.channels[0]):
            array_2 = []
            for i in range(self.size[0]):
                array_2.append(list(map(float, array_1[i].split())))
                if len(array_2[i]) != self.size[0]:
                    exit(f'input.txt rows!=cols on {i} row')
            current_array.append(array_2)
        for i in range(len(self.size) - 1):
            if self.operations[i] == 0:
                current_array = self.use_kernel(self, i, current_array)
                # use df
            elif self.operations[i] == 1:
                for j in range(self.channels[i]):
                    current_array[j] = self.use_pooling(self, i, current_array[j])
            elif self.operations[i] == 2:
                current_array = self.use_mp(self, i, i_for_mp, current_array)
                i_for_mp += 1
            elif self.operations[i] == 3:
                current_array = self.use_flatten(self, current_array)
            else:
                exit(f'unknown operation on the {i} layer')
        return current_array


"""
shortnames:
MP - multilayer perceptron
config.txt structure:
0 line: number of neurons in row/col on every layer, then num of neurons on every layer of MP
1 line: num of channels on every layer, except MP 
2 line: operation on layer (0 - convolution; 1 - pooling; 2 - MP)
3 line: length / width of kernel ( 2x2 type only 2), except MP
4 line: for convolution one side of padding( 0 - no pad); for pooling 0 - max; 1 - avg
5 line: activate function (1 - sigmoid; 2 - modRELu; 3 - th) for convolution and MP; for padding - stride num
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
Net.__init__(Net, main_size, main_channels, main_operations, main_kernel_size, main_pad_n_pool_func,
             main_act_func_choices)
print(Net.forward_feed(Net))

'''!!!!!!!!
ACTFUNCTIONS
'''
