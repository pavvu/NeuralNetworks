import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os, struct
from array import array as pyarray

# Reading input from the images
def getLabelsArray(fname, size_img, output_dim):
    flbl = open(fname, 'rb')
    magic_nr, size_label = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    desired_label = np.zeros((size_img, output_dim), dtype=np.int)
    for i in range(size_label):
        desired_label[i][lbl[i]] = 1

    return desired_label

def getImagesArray(fname):
    fimg = open(fname, 'rb')
    magic_nr, size_img, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    img = np.asarray(img).reshape(size_img, rows*cols)
    fimg.close()
    return img, size_img

## Global Variables
x, size_img = getImagesArray("train-images.idx3-ubyte")
d = getLabelsArray("train-labels.idx1-ubyte", size_img, 10)
x = x.T

testImg, test_size_img  = getImagesArray("t10k-images.idx3-ubyte")
test_d = getLabelsArray("t10k-labels.idx1-ubyte", test_size_img, 10)
testImg = testImg.T
testExamples = testImg.shape[1]

#x = np.random.randn(784,5)
trainingExamples = 15000
print ("trainingExamples ", trainingExamples)
#d = np.random.randn(10,10)
fillOnes = np.array(np.ones(trainingExamples))
v = []
neuronsInHiddenLayer = 75
wInPut = np.random.randn(neuronsInHiddenLayer,784)
wOutPut = np.random.randn(10,neuronsInHiddenLayer)
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
    #op1 = v
    return op1

def getOutPut2(v):
    global op2
    # op1 = np.tanh(v)
    op2 = np.tanh(v)
    return op2

def getVdash(outPut1):
     global wOutPut, vDash
     #print ("wOutput ", wOutPut)
     vDash = wOutPut.dot(outPut1)
     return vDash

def getPhiDash(v):
    derivative = 1 - np.power(np.tanh(v), 2)
    return derivative
    #return v.fill(1)

# to find index of maximum value in a matrix of size 10X1
# is required to build computed output, which will contain 1 at maximumvalue and 0 at other places
def findMaxIndex (tempV):
    maxNumber = -100000
    maxIndex = -1
    for j in range (0,10):
        if (tempV[j][0] > maxNumber):
            maxNumber = tempV[j][0]
            maxIndex = j;
    return maxIndex

def getVectorWithOneAtIndex(maxIndex):
    tempV =np.random.rand(10,1)
    tempV.fill(0)
    tempV[maxIndex][0]=1
    return tempV

def getMSE(dForInPutI, op):
    mse = 0.0
    for i in range(0,10):
        mse = mse + math.pow((dForInPutI[0][i] - op[0][i]) ,2)
    mse = mse/2

def isCorrectlyClassified(forInPutI, data, labels, debug=False):
    currx = data[:, forInPutI]
    v = wInPut.dot(currx)
    op1 = getOutPut1(v)
    vDash = getVdash(op1)
    op2 =  getOutPut2(vDash)
    op2 = op2.reshape(10,1)
    maxIndex = findMaxIndex(op2)
    op2 = getVectorWithOneAtIndex(maxIndex)
    dForInPutI = labels[forInPutI][:]
    op2 = op2.T
    mse  = getMSE(dForInPutI, op2)
    if (not (dForInPutI == op2).all()):
       return mse, False
    return mse, True


def getError(forInPutI, debug):
    global vDash, op1, v, computedOP, x
    currx = x[:, forInPutI]
    v = wInPut.dot(currx)
    op1 = getOutPut1(v)
    vDash = getVdash(op1)
    op2 = getOutPut2(vDash)

    #computedOP.append(op2)
    # need to change desired output
    dForInPutI = d[forInPutI][:]
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
    delta = error #* phiDashOfVDash
    delta = delta.reshape(10,1)
    op1 = op1.reshape(1,neuronsInHiddenLayer)
    deltaForWOutPut = delta * op1
    deltaForWInPut = np.random.rand(neuronsInHiddenLayer,784)
    currx = x[:, forInPutI]
    #deltaForWInPut = (getPhiDash(v) * delta).T.dot(wOutPut)
    phiDashOfV  = getPhiDash(v).reshape(neuronsInHiddenLayer,1)
    deltaForWInPut = (wOutPut.T).dot(delta) #* phiDashOfV
    #print ("phiDashOfV ", phiDashOfV)
    deltaForWInPut = deltaForWInPut * currx
    shape = deltaForWInPut.shape
    #print ("shape ", shape)
    # for i in range (0, 784):
    #     for j in range(0, 784):
    #         delta1 = delta * getPhiDash(v)
    #         deltaForWInPut[i][j] =  delta1 * currx[j]

    return deltaForWOutPut, deltaForWInPut

converged = False
learningRate = 0.003
epoch = 0
prev_er = 1
epochCount = []
noOfTrainErros = []
noOfTestErros = []
MSEinEpoch = []
while(not converged):
    start_time = time.time()
    epoch +=1
    epochCount.append(epoch)

    shuffled_index = list(range(0, trainingExamples))
    random.shuffle(shuffled_index)
    wInPutPrevious = wInPut
    wOutPutPrevious = wOutPut
    shuffled_index = list(range(0, trainingExamples))
    random.shuffle(shuffled_index)
    for forInPutI in  ( shuffled_index):
        deltaForWOutPut, deltaForWInPut = getUpdates(forInPutI, False)
        wInPut = wInPut + learningRate * deltaForWInPut
        wOutPut = wOutPut + learningRate * deltaForWOutPut
    print("epoch ", epoch)

    erros = 0.0
    computedOP = []
    mse = 0.0
    for index in range(0, trainingExamples):
        tempMse, correctlyClassified = isCorrectlyClassified(index, x, d)
        mse = mse + tempMse
        if (not correctlyClassified):
            erros += 1
    er = erros / trainingExamples
    noOfTrainErros.append(erros)
    mse = mse/trainingExamples
    MSEinEpoch.append(mse)

    if (er >= prev_er):
        learningRate = learningRate * 0.9
        wInPut = wInPutPrevious
        wOutPut = wOutPutPrevious

    print("error rate ", er, " learning rate ", learningRate)
    prev_er = er

    #Testing images
    if (not converged) :
        testErros = 0.0

        for index in range(0, testExamples):
            if (not isCorrectlyClassified(index, testImg, test_d)):
                testErros += 1
        testErrorRate = testErros / testExamples
        noOfTestErros.append(testErros)
        print ("test error rate ", testErrorRate)
        if(testErrorRate < 0.1 or epoch > 24):
            plt.plot(epochCount, noOfTrainErros, 'b')
            plt.plot(epochCount, noOfTestErros, 'r')
            plt.show()
            plt.plot(epochCount, MSEinEpoch, 'r')
            plt.show()
            converged = True


    print("tim taken for this epoch ", time.time()-start_time)
