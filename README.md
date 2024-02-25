Closed, because CNN v2 is better
"""
shortnames:
MP - multilayer perceptron
config.txt structure:
1 line: number of neurons in row/col on every layer, then num of neurons on every layer of MP
2 line: num of channels on every layer, except MP 
3 line: operation on layer (0 - convolution; 1 - pooling; 2 - MP; 3 - flattening (to work with MP))
4 line: length / width of kernel ( 2x2 type only 2), except MP
5 line: for convolution one side of padding( 0 - no pad); for pooling 0 - max; 1 - avg
6 line: activate function (1 - sigmoid; 2 - modRELu; 3 - th) for convolution and MP; for padding - stride num
7 line: using mode (0 - if you want to use; 1 - if you want to learn)

"""
