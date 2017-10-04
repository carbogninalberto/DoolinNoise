# DoolinNoise
#JE7IO792LHC3VYS5

## ----------------------- Part 1 ---------------------------- ##
import numpy as np
import urllib.request
from numpy import genfromtxt
from tinydb import TinyDB, Query
import sqlite3
#!/usr/bin/python3

#import PyMySQL

#import for matplotlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

import matplotlib.pyplot as plt



#import tkinter
import tkinter as tk
from tkinter import ttk

import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

index = input("Tell me stock Acronym UPPER LETTER: ")
if (index != ''):
    file_name = index + '5min.csv'
    stock_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + index + '&interval=5min&apikey=JE7IO792LHC3VYS5&datatype=csv'
    with urllib.request.urlopen(stock_url, context=ctx) as u, open('database/'+file_name, 'wb') as f:
        f.write(u.read())
if (index != ''):
    file_name = index + '5min_SMA.csv'
    stock_url = 'https://www.alphavantage.co/query?function=SMA&symbol=' + index + '&interval=weekly&time_period=10&series_type=close&apikey=JE7IO792LHC3VYS5&datatype=csv'
    with urllib.request.urlopen(stock_url, context=ctx) as u, open('database/'+file_name, 'wb') as f:
        f.write(u.read())




#https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&outputsize=full&apikey=JE7IO792LHC3VYS5
#import url of stock
#urllib.request.urlretrieve('http://www.google.com/finance/historical?q=NASDAQ:AAPL&ei=pkaCWZnfMZfRUqDlgaAL&output=csv', 'history.csv')

#urllib.request.urlretrieve('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&outputsize=full&apikey=JE7IO792LHC3VYS5', '/database/aapl15min.csv', context=context)

#database settings

#stock = TinyDB('database/AAPL5min.csv')
#Data = Query()
#res = stock.search(Data['Meta Data'].exists())
#res = stock.all()
#print(res)
# Noise Prevision Index - [Previsione:float, Errore:float]
# mi dice quanto mi sbaglio mediamente
# indice di misurazione di efficienza dell'algoritmo non inserito
npi_db = TinyDB('database/npi.json')

#Day Based Index - [Giorno:string, DBI:float]
# mi dice l'errore medio in un determinato giorno della settimana (Lun-Ven)
dbi_db = TinyDB('database/dbi.json')

# Month Based Index - [Mese:String, CrescitaPercentuale:float]
# mi dice la crescita massima attesa in quel mese
mbi_db = TinyDB('database/mbi.json')

#data elaboration
mydata = genfromtxt('data.csv', delimiter=',')
mydata = np.delete(mydata, 0, 1)
volume = mydata
mydata = np.delete(mydata, 4, 1)
nrow = mydata.shape[0]-1
outputArr = np.delete(mydata, nrow, 0)
outputArr = np.delete(outputArr, 0, 1)
outputArr = np.delete(outputArr, 0, 1)
outputArr = np.delete(outputArr, 0, 1)
mydata = np.delete(mydata, 0, 0)

volume = np.delete(volume, 0, 1)
volume = np.delete(volume, 0, 1)
volume = np.delete(volume, 0, 1)
volume = np.delete(volume, 0, 1)
volume = np.delete(volume, 0, 0)
#volume = volume.transpose()[0]

mydata = np.array(mydata, dtype=float)
outputArr = np.array(outputArr, dtype=float)
volume = np.array(volume, dtype=float)

print(volume)

#print mydata
#print "\n output array is: \n"
#print outputArr
# X = (hours sleeping, hours studying), y = Score on test
#X = np.array(([100.2, 102, 97, 101.5], [98, 103, 95, 96], [98, 115, 58, 73]), dtype=float)
#y = np.array(([107], [101.5], [96]), dtype=float)

X = mydata
y = outputArr
X_v = volume

#plt.plot(y)
#plt.ylabel('X values')
#plt.show()

max_x = np.amax(X)
min_x = np.amin(X)
max_xv = np.amax(X_v)
min_xv = np.amin(X_v)

# Normalize
recover = np.amax(y)-np.amin(y)
min_sum = np.amin(y)
X = (X-np.amin(X))/(np.amax(X)-np.amin(X))
X_v = (X_v-np.amin(X_v))/(np.amax(X_v)-np.amin(X_v))
y = (y-np.amin(y))/(np.amax(y)-np.amin(y))

print (X.shape, " ", X_v.shape)
X = np.append(arr=X, values=X_v, axis=1)

print (X)
print (y)




#AAPL data
#print np.amax(mydata, axis = 0)
#print np.amax(outputArr, axis = 0)


#recover = np.amax(outputArr, axis = 0)
#X = mydata/np.amax(mydata, axis = 0)
#y = outputArr/np.amax(outputArr, axis = 0)

#recover = 107

def noise_prevision_index(file):
    return True


## ----------------------- Part 5 ---------------------------- ##
## 21 - 17 perfetto per la previsione giornaliera
## 17 - 21 5 mins trading - forse
class Neural_Network(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 5
        self.outputLayerSize = 1
        self.hiddenLayerSize1 = 21 #21
        self.hiddenLayerSize2 = 17 #17

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize1)
        self.W2 = np.random.randn(self.hiddenLayerSize1,self.hiddenLayerSize2)
        self.W3 = np.random.randn(self.hiddenLayerSize2,self.outputLayerSize)


    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)

        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W3 for a given X and y:
        self.yHat = self.forward(X)

        delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T, delta4)

        delta3 = np.dot(delta4, self.W3.T)*self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2, dJdW3

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W3 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W3 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize1 * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize1))

        W2_end = W1_end + self.hiddenLayerSize1*self.hiddenLayerSize2
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize1 , self.hiddenLayerSize2))

        W3_end = W2_end + self.hiddenLayerSize2*self.outputLayerSize
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayerSize2, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

## ----------------------- Part 6 ---------------------------- ##
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 10000, 'disp' : True} #BFGS
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)



        self.N.setParams(_res.x)

        self.optimizationResults = _res


if __name__ == "__main__":
    NN = Neural_Network()
    #print NN.W1
    #print "Prima della fase di train \n"
    #print NN.forward(X)
    #print "####################"
    print (NN.forward(X))
    Trainer = trainer(NN)

    Trainer.train(X, y)
    print ("Post fase di train \n")
    #print NN.forward(X)
    #print "####################"
    print (NN.forward(X) * recover + min_sum)

    save_ = input("Want to save Weights?[y/n] ")
    if (save_ == 'Y' or save_ == 'y'):
        np.savetxt("w1.out", NN.W1, delimiter=',')
        np.savetxt("w2.out", NN.W2, delimiter=',')
        np.savetxt("w3.out", NN.W3, delimiter=',')
        print ("weights exported sucessfully!")
    make_prediction_ = input("Prediction based on actual Training Session?[y/n] ")
    if (make_prediction_ == 'y' or make_prediction_ == 'Y'):
        predValues = input("Open, High, Low, Close, Volume: ")
        predValues = predValues.split(',')
        stockOpenPrice = predValues[0]
        stockHighPrice = predValues[1]
        stockLowPrice = predValues[2]
        stockClosePrice = predValues[3]
        stockVolume = predValues[4]
        newX = np.array(([stockOpenPrice, stockHighPrice, stockLowPrice, stockClosePrice]), dtype = float)
        newX_v = np.array([stockVolume], dtype=float)

        if (max_x < np.amax(newX)):
            max_x = np.amax(newX)
        if (min_x > np.amin(newX)):
            min_x = np.amin(newX)
        if (max_xv < np.amax(newX_v)):
            max_xv = np.amax(newX_v)
        if (min_xv > np.amin(newX_v)):
            min_xv = np.amin(newX_v)

        #newX = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        #X_v = (X_v-np.amin(X_v))/(np.amax(X_v)-np.amin(X_v))

        newX = (newX-min_x)/(max_x - min_x)
        newX_v = (newX_v-min_xv)/(max_xv - min_xv)
        newX = np.append(arr=newX, values=newX_v, axis=0)
        print(newX)

        print (NN.forward(newX) * recover + min_sum)

    #Trainer = trainer()
