# DoolinNoise


## ----------------------- Part 1 ---------------------------- ##
import numpy as np
from numpy import genfromtxt
#data elaboration
mydata = genfromtxt('data.csv', delimiter=',')
mydata = np.delete(mydata, 0, 1)
mydata = np.delete(mydata, 4, 1)
nrow = mydata.shape[0]-1
outputArr = np.delete(mydata, nrow, 0)
outputArr = np.delete(outputArr, 0, 1)
outputArr = np.delete(outputArr, 0, 1)
outputArr = np.delete(outputArr, 0, 1)
mydata = np.delete(mydata, 0, 0)

mydata = np.array(mydata, dtype=float)
outputArr = np.array(outputArr, dtype=float)
mydata = mydata
outputArr = outputArr
print outputArr
print mydata
#print mydata
#print "\n output array is: \n"
#print outputArr
# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([100.2, 102, 97, 101.5], [98, 103, 95, 96], [98, 115, 58, 73]), dtype=float)
y = np.array(([107], [101.5], [96]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/107

#AAPL data
#print np.amax(mydata, axis = 0)
#print np.amax(outputArr, axis = 0)
recover = np.amax(outputArr, axis = 0)
X = mydata/np.amax(mydata, axis = 0)
y = outputArr/np.amax(outputArr, axis = 0)
#recover = 107


## ----------------------- Part 5 ---------------------------- ##

class Neural_Network(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 4
        self.outputLayerSize = 1
        self.hiddenLayerSize1 = 21
        self.hiddenLayerSize2 = 17

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

        options = {'maxiter': 200, 'disp' : True}
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
    print NN.forward(X) * recover

    Trainer = trainer(NN)

    Trainer.train(X, y)
    print "Post fase di train \n"
    #print NN.forward(X)
    #print "####################"
    print NN.forward(X) * recover

    save_ = raw_input("Wanna Save Weights?[Y/n] ")
    if (save_ == 'Y' or save_ == 'y'):
        print "weights exported sucessfully!"

    print "Prediction based on actual Training Session"
    stockOpenPrice = raw_input("Stock Open Price: ")
    stockHighPrice = raw_input("Stock High Price: ")
    stockLowPrice = raw_input("Stock Low Price: ")
    stockClosePrice = raw_input("Stock Close Price: ")
    newX = np.array(([stockOpenPrice, stockHighPrice, stockLowPrice, stockClosePrice]), dtype = float)
    newX = newX/np.amax(newX, axis=0)
    print NN.forward(newX) * recover

    #Trainer = trainer()
