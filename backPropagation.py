import math
import random
import numpy as np
import matplotlib.pyplot as plt

## Global Variables
trainingExamples = 300
x = []
##net value at input layer
v = []
wInPut = []
wOutPut = []
d = []
##output from input layer
op1 = []
##net value at output layer
vDash = []
computedOP = []

##generating inout and calculating desired output from given function
def generateXandD():
    global x, d
    x = np.random.uniform(0, 1, size=trainingExamples)
    x = np.sort(x)
    vip = np.random.uniform(-0.1, 0.1, size=trainingExamples)
    d = np.add( np.sin(20*x), 3*x)
    d = np.add(d, vip)

def plotXvsD(computed):
    global x, d
    for i in range(0,trainingExamples):
        plt.scatter(x[1][i], computed[i], color='b', marker = '*')
    for i in range(0, trainingExamples):
        plt.scatter(x[1][i], d[i], color='r', marker='*')
    plt.show()

def initializeWeights():
    global  wInPut, wOutPut
    wInPut = np.random.uniform(-2,2,[24,2])
    wOutPut = np.random.uniform(-2,2,[1,25])

#adding row of 1 to input vector for biases
def buildInputVector():
    global x
    fillzeros = np.array(np.zeros(trainingExamples))
    fillOnes = np.array(np.ones(trainingExamples))
    x = np.vstack([ fillOnes, x])
    #print ("x values ", x)

def getV(forInputI):
    global wInPut, x
    i = forInputI
    #print("Winput ", wInPut)
    #taking the current input
    currx = x[:,i]
    return wInPut.dot(currx)

def getOutPut1(v):
    global op1
    op1 = np.tanh(v)
    op1 = np.append(1,op1)
    return op1

def getVdash(outPut1):
     global wOutPut, vDash
     #print ("wOutput ", wOutPut)
     vDash = wOutPut.dot(outPut1)
     return vDash

def getPhiDash(v):
    tanhValue = np.tanh(v)
    derivative = 1 - math.pow(tanhValue, 2)
    return derivative

def getError(forInPutI):
    global vDash, op1, v, computedOP
    v = getV(forInPutI)
    #print("printing v1", v, "shape ", v.shape)
    op1 = getOutPut1(v)
    #print("printing op1", op1, " shape ", op1.shape)
    vDash = getVdash(op1)
    #print("vdash ", vDash)
    op2 = np.tanh(vDash)
    op2 = op2[0]
    computedOP.append(op2)
    #print("op2 ", op2)
    dForInPutI = d[forInPutI]
    #print("desired op for input", forInPutI, " : ", dForInPutI)
    #print("dForInPutI ", dForInPutI)
    error = (dForInPutI - op2)
    return error


def getUpdates(forInPutI):
    global  vDash, op1, v, x, wOutPut
    error = getError(forInPutI)
    phiDashOfV = getPhiDash(vDash)
    #print ("error, ", error, " error value ", error[0], "phidash ", phiDashOfV, " value ", phiDashOfV[0])
    delta = error * phiDashOfV
    #print ("delta ", delta)
    deltaForWOutPut = delta * op1
    ## Delta for input layer weights
    deltaForWInPut = np.random.rand(24,2)
    currx = x[:, forInPutI]
    for index in range (0, 24):
        #print("wOutPut[0][index] ", wOutPut[0][index], "getPhiDash(v[index] ", getPhiDash(v[index]))
        delta1 = delta * wOutPut[0][index+1] * getPhiDash(v[index])
        #print("C0 ", currx[0])
        #print("C1 ", currx[1])
        deltaForWInPut[index][0] =  delta1 * currx[0]
        deltaForWInPut[index][1] =  delta1 * currx[1]

    return deltaForWOutPut, deltaForWInPut


generateXandD()
#computeDesiredOP()
initializeWeights()
buildInputVector()

converged = False
learningRate = 0.06

epoch = 0
while(not converged):
    epoch +=1
    MSE = 0.0
    computedOP = []
    for index in range(0, trainingExamples):
        error = getError(index)
        MSE = MSE + math.pow(error, 2)
    MSE = MSE/trainingExamples

    if(MSE < 0.5):
        converged = True

    if (epoch%100 ==0):
        plotXvsD(computedOP)
        print("epoch ", epoch, " MSE ", MSE)


    for forInPutI in range(0, trainingExamples):
        deltaForWOutPut, deltaForWInPut = getUpdates(forInPutI)
        wInPut = wInPut + learningRate * deltaForWInPut
        wOutPut = wOutPut + learningRate * deltaForWOutPut

#plotXvsD()
# w1, w2 = getUpdates(1)
# print( "delta for woutput", w1)
# print( "delta for winput", w2)