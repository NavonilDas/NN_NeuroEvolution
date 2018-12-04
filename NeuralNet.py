import random
import math
import numpy as np
import pickle
import os
from ActivationFunction import *

class NeuralNet:
    # Randomly Make Values Negative
    def Randomize(self,ar):
        r = ar.shape[0]
        c = ar.shape[1]
        for i in range(r):
            for j in range(c):
                if(random.uniform(0,1) < 0.4):
                    ar[i][j] *= -1
    
    def __init__(self,inp,nhl,nop):
        self.fitness = 0                    # Fitness of the Neural net
        self.nhl = nhl                      # Number of Hiddden layer
        self.nop = nop                      # Number of Output
        self.inp = inp                      # Number of Input
        self.mutationRate = 0.01            # The Mutation Rate

        self.inpvec = np.random.rand(inp,1) # Input Vector
        self.wIH = np.random.rand(nhl,inp)  # Weight Matrix Between Input and Hidden Layer
        self.iBias = np.random.rand(nhl,1)  # Hidden Layer Bias
        
        self.wHO = np.random.rand(nop,nhl)  # wight matrix between hidden layer and output
        self.oBias = np.random.rand(nop,1)  # Output bias
        self.out = np.random.rand(nop,1)    # Output Vector

        self.hl =  np.random.rand(nhl,1)    # Hidden Layer
        self.af = ReLU                      # Activation Function

        # Make Some Values Negative
        self.Randomize(self.inpvec)
        self.Randomize(self.wIH)
        self.Randomize(self.iBias)
        self.Randomize(self.wHO)
        self.Randomize(self.oBias)
            
    def setFitness(self,val):
        self.fitness = val
    
    def setActivationFunction(self,ind):
        if ind == 0:self.af = ReLU
        elif ind == 1:self.af = Sigmoid
        elif ind == 2:self.af = Tanh
        elif ind == 3:self.af = BinaryStep
        elif ind == 4:self.af = RandomizedRelu
        elif ind == 5:self.af = LeakyReLU

    def fitnessMultiplier(self,val):
        self.fitness *= val

    def setMutationRate(self,val):
        self.mutationRate = val
    
    def setInput(self,val):
        for i in range(self.inp):
            self.inpvec[i][0] = val[i]
    
    def calcHiddenLayers(self):
        self.hl = np.dot(self.wIH,self.inpvec)
        for i in range(self.nhl):
            self.hl[i] = self.af(self.hl[i]+self.iBias[i])
    
    def calcOpLayers(self):
        self.out = np.dot(self.wHO,self.hl)
        for i in range(self.nop):
            self.out[i] = self.af(self.out[i]+self.oBias[i])

    # Crossover and Mutation
    # http://www.obitko.com/tutorials/genetic-algorithms/crossover-mutation.php
    
    #### Reproduce and Mutate wIH
    def __rWIH(self,a,b,partner):
        if a < b:a,b = b,a        
        for i in range(self.nhl*self.inp):
            if i>=b and i<=a:
                self.wIH[int(i/self.inp)][i%self.inp] = partner.wIH[int(i/self.inp)][i%self.inp]
            if(random.uniform(0,1) <= self.mutationRate):
                self.wIH[int(i/self.inp)][i%self.inp] = random.uniform(-1,1)
    #### Reproduce and Mutate wHO
    def __rWHO(self,a,b,partner):
        if a < b: a,b = b,a
        for i in range(self.nop*self.nhl):
            if i>=b and i<=a:
                self.wHO[int(i/self.nhl)][i%self.nhl] = partner.wHO[int(i/self.nhl)][i%self.nhl]
            if(random.uniform(0,1) <= self.mutationRate):
                self.wHO[int(i/self.nhl)][i%self.nhl] = random.uniform(-1,1)
    #### Reproduce and Mutate output Bias
    def __rOBIAS(self,a,b,partner):
        if a < b: a,b = b,a
        for i in range(self.nop):
            if i>=b and i<=a:
                self.oBias[i] = partner.oBias[i]
            if(random.uniform(0,1) <= self.mutationRate):
                self.oBias[i] = random.uniform(-1,1)
    #### Reproduce and Mutate input Bias
    def __rIBIAS(self,a,b,partner): 
        if a < b: a,b = b,a
        for i in range(self.nhl):
            if i>=b and i<=a:
                self.iBias[i] = partner.iBias[i]
            if(random.uniform(0,1) <= self.mutationRate):
                self.iBias[i] = random.uniform(-1,1)
    #### Get Two Random Points
    def __getTwoRandomPoints(self,Max):
        a = random.randint(0,Max)
        b = random.randint(0,Max)
        while a==b:
            a = random.randint(0,Max)
            b = random.randint(0,Max)
        return a,b

    # Reproduce using Single crossover
    def ReproduceSC(self,partner):
        cp = random.randint(1,self.nhl*self.inp)
        self.__rWIH(0,cp,partner)
        cp = random.randint(1,self.nop*self.nhl)
        self.__rWHO(0,cp,partner)
        cp = random.randint(1,self.nop)
        self.__rOBIAS(0,cp,partner)
        cp = random.randint(1,self.inp)
        self.__rIBIAS(0,cp,partner)
        
    
    ### TODO:
    ### CHECK BOUNDS , FIX for One output

    
    # Reproduce using Two Point Crossover
    def ReproduceTC(self,partner):
        a,b = self.__getTwoRandomPoints(self.nhl*self.inp)
        self.__rWIH(a,b,partner)
 
        a,b = self.__getTwoRandomPoints(self.nhl*self.inp)
        self.__rWHO(a,b,partner)

        a,b = self.__getTwoRandomPoints(self.nop)
        self.__rOBIAS(a,b,partner)
        
        a,b = self.__getTwoRandomPoints(self.nhl)
        self.__rIBIAS(a,b,partner)

    def SaveText(self,ind):
        if not os.path.isdir("saved"):
            os.mkdir("saved")
        with open('saved/wIH'+str(ind)+'.dat',"wb") as fout:
            pickle.dump(self.wIH,fout,pickle.HIGHEST_PROTOCOL)
        with open('saved/wHO'+str(ind)+'.dat',"wb") as fout:
            pickle.dump(self.wHO,fout,pickle.HIGHEST_PROTOCOL)
        with open('saved/oBias'+str(ind)+'.dat',"wb") as fout:
            pickle.dump(self.oBias,fout,pickle.HIGHEST_PROTOCOL)
        with open('saved/iBias'+str(ind)+'.dat',"wb") as fout:
            pickle.dump(self.iBias,fout,pickle.HIGHEST_PROTOCOL)
    
    # Reading Save Weight and Bias
    def ReadText(self,ind):
        if not os.path.isdir("saved"):
            raise Exception("Saved Directory Not Found")
        if os.path.isfile('saved/wIH'+str(ind)+'.dat') or os.path.isfile('saved/wHO'+str(ind)+'.dat') or os.path.isfile('saved/oBias'+str(ind)+'.dat') or os.path.isfile('saved/iBias'+str(ind)+'.dat'):
            raise Exception("File Not Found")

        with open('saved/wIH'+str(ind)+'.dat',"rb") as fin:
            self.wIH = pickle.load(fin)

        with open('saved/wHO'+str(ind)+'.dat',"rb") as fin:
            self.wHO = pickle.load(fin)

        with open('saved/oBias'+str(ind)+'.dat',"rb") as fin:
            self.oBias = pickle.load(fin)
        
        with open('saved/iBias'+str(ind)+'.dat',"rb") as fin:
            self.iBias = pickle.load(fin)