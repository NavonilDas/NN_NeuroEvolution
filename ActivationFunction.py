import math

#   https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

def ReLU(x):
    return max(0,x)

def Sigmoid(x):
    return 1/(1+math.exp(-x))

def Tanh(x):
    return math.tanh(x)

def BinaryStep(x):
    if x >= 0: return 1
    else:return 0

def RandomizedRelu(a,x):
    if x >= 0:
        return x
    else:
        return a*x

def LeakyReLU(x):
    RandomizedRelu(0.01,x)