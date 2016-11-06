import math
import random
import numpy as np
import matplotlib.pyplot as plt

## Global Variables
trainingExamples = 5
x = np.random.randn(784,5)
d = np.random.randn(10,10)
fillOnes = np.array(np.ones(trainingExamples))
v = []
wInPut = np.random.randn(784,784)
wOutPut = np.random.randn(10,784)
op1 = []
vDash = []
computedOP = []

def plotXvsD(computed):
    global x, d
    for i in range(0,trainingExamples):
        plt.scatter(x[1][i], computed[i], color='b', marker = '*')
    for i in range(0, trainingExamples):
        plt.scatter(x[1][i], d[i], color='r', marker='*')
    plt.show()

def getOutPut1(v):
    global op1
    op1 = np.tanh(v)
    return op1

def getVdash(outPut1):
     global wOutPut, vDash
     #print ("wOutput ", wOutPut)
     vDash = wOutPut.dot(outPut1)
     return vDash

def getPhiDash(v):
    derivative = 1 - np.power(np.tanh(v), 2)
    return derivative

def getError(forInPutI, debug):
    global vDash, op1, v, computedOP, x
    currx = x[:, forInPutI]
    v = wInPut.dot(currx)
    op1 = getOutPut1(v)
    vDash = getVdash(op1)
    op2 = np.tanh(vDash)

    computedOP.append(op2)
    # need to change desired output
    dForInPutI = d[forInPutI]
    error = (dForInPutI - op2)
    if debug:
        print("desired op for input", forInPutI, " : ", dForInPutI)
        print("dForInPutI ", dForInPutI)
        print("printing v1", v, "shape ", v.shape)
        print("printing op1", op1, " shape ", op1.shape)
        print("op2 ", op2)
        print("vdash ", vDash)
    return error


def getUpdates(forInPutI, debug):
    global  vDash, op1, v, x, wOutPut
    error = getError(forInPutI, False)
    phiDashOfVDash = getPhiDash(vDash)
    delta = error * phiDashOfVDash
    delta = delta.reshape(10,1)
    op1 = op1.reshape(1,784)
    deltaForWOutPut = delta * op1
    deltaForWInPut = np.random.rand(784,784)
    currx = x[:, forInPutI]
    #deltaForWInPut = (getPhiDash(v) * delta).T.dot(wOutPut)
    deltaForWInPut = (wOutPut.T).dot(delta) * getPhiDash(v).reshape(784,1)
    shape = deltaForWInPut.shape
    print ("shape ", shape)
    # for i in range (0, 784):
    #     for j in range(0, 784):
    #         delta1 = delta * getPhiDash(v)
    #         deltaForWInPut[i][j] =  delta1 * currx[j]

    return deltaForWOutPut, deltaForWInPut

converged = False
learningRate = 0.05
epoch = 0
while(not converged):
    epoch +=1
    MSE = 0.0
    computedOP = []
    # for index in range(0, trainingExamples):
    #     error = getError(index, False)
    #     MSE = MSE + math.pow(error, 2)
    # MSE = MSE/trainingExamples
    #
    # if(MSE < 0.01):
    #     converged = True
    #     plotXvsD(computedOP)
    #     #plotXvsD(computedOP)
    #
    # if (epoch%300 ==0):
    #     print("epoch ", epoch, " MSE ", MSE)

    # if(epoch%3000 == 0 and epoch > 0):
    #     learningRate *= 0.9

    shuffled_index = list(range(0, trainingExamples))
    random.shuffle(shuffled_index)
    for forInPutI in range (0, trainingExamples):
        deltaForWOutPut, deltaForWInPut = getUpdates(forInPutI, False)
        wInPut = wInPut + learningRate * deltaForWInPut
        wOutPut = wOutPut + learningRate * deltaForWOutPut
