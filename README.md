Current condition:
Supports only 2 layers

Open config.txt and change its structure

Structure of config.txt:
0 line: number of neurons in each layer
1 line: activate function (1 - sigmoid; 2 - modRELu; 3 - th)
2 line: using mode (0 - using from console; 1 - using from file; 2 - learning from datafile)

Choose the mode you're going to use
Open main.py
If you want to learn, edit learn.txt

learn.txt structure:
1,3,5... Row is array of true results
2,4,6... Row is array of input neurons

After editing num_of_learns and num_of_laps
START, GOOD LUCK!
