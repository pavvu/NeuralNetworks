import math
import random
import numpy as np
import matplotlib.pyplot as plt

## Global Variables
trainingExamples = 300
x = np.sort(np.random.uniform(0, 1, size=trainingExamples))
d = np.add(np.add( np.sin(20*x), 3*x), np.random.uniform(-0.1, 0.1, size=trainingExamples))
fillOnes = np.array(np.ones(trainingExamples))
x = np.vstack([ fillOnes, x])
v = []
wInPut = np.random.randn(24,2)
wOutPut = np.random.randn(1,25)
op1 = []
vDash = []
computedOP = []

# for i in range(0, trainingExamples):
#     plt.scatter(x[1][i], d[i], color='r', marker='o')
# plt.show()

def plotXvsD(computed):
    global x, d
    for i in range(0,trainingExamples):
        plt.scatter(x[1][i], computed[i], color='b', marker = '*')
    for i in range(0, trainingExamples):
        plt.scatter(x[1][i], d[i], color='r', marker='+')
    plt.show()

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
    derivative = 1 - math.pow(np.tanh(v), 2)
    return derivative

def getError(forInPutI, debug):
    global vDash, op1, v, computedOP, x
    currx = x[:, forInPutI]
    v = wInPut.dot(currx)
    op1 = getOutPut1(v)
    vDash = getVdash(op1)
    op2 = vDash[0]
    computedOP.append(op2)
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
    phiDashOfV = 1
    delta = error * phiDashOfV
    deltaForWOutPut = delta * op1
    deltaForWInPut = np.random.rand(24,2)
    currx = x[:, forInPutI]
    for index in range (0, 24):
        delta1 = delta * wOutPut[0][index+1] * getPhiDash(v[index])
        deltaForWInPut[index][0] =  delta1 * currx[0]
        deltaForWInPut[index][1] =  delta1 * currx[1]
    return deltaForWOutPut, deltaForWInPut

converged = False
learningRate = 0.05
epoch = 0
MSEInEpoch = []
epochs = []
while(not converged):
    epoch +=1
    epochs.append(epoch)
    MSE = 0.0
    computedOP = []
    for index in range(0, trainingExamples):
        error = getError(index, False)
        MSE = MSE + math.pow(error, 2)
    MSE = MSE/trainingExamples
    MSEInEpoch.append(MSE)

    if(MSE < 0.01):
        converged = True
        plotXvsD(computedOP)
        plt.axis([0, epoch, 0, 1.5])
        plt.plot(epochs, MSEInEpoch, 'b')
        plt.xlabel("EpochCount")
        plt.ylabel("MSE")
        plt.show()
        #plotXvsD(computedOP)

    if (epoch%300 ==0):
        print("epoch ", epoch, " MSE ", MSE)

    # if(epoch%3000 == 0 and epoch > 0):
    #     learningRate *= 0.9

    shuffled_index = list(range(0, trainingExamples))
    random.shuffle(shuffled_index)
    for forInPutI in range (0, trainingExamples):
        deltaForWOutPut, deltaForWInPut = getUpdates(forInPutI, False)
        wInPut = wInPut + learningRate * deltaForWInPut
        wOutPut = wOutPut + learningRate * deltaForWOutPut
