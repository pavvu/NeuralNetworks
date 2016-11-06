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
#x = np.random.randn(784,5)
trainingExamples = 200
print ("trainingExamples ", trainingExamples)
#d = np.random.randn(10,10)
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

def isCorrectlyClassified(forInPutI, debug=False):
    global  x
    currx = x[:, forInPutI]
    v = wInPut.dot(currx)
    op1 = getOutPut1(v)
    vDash = getVdash(op1)
    op2 =  getOutPut2(vDash)
    op2 = op2.reshape(10,1)
    maxIndex = findMaxIndex(op2)
    op2 = getVectorWithOneAtIndex(maxIndex)
    dForInPutI = d[forInPutI][:]
    op2 = op2.T
    if (not (dForInPutI == op2).all()):
       return False
    return True


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
    delta = error * phiDashOfVDash
    delta = delta.reshape(10,1)
    op1 = op1.reshape(1,784)
    deltaForWOutPut = delta * op1
    deltaForWInPut = np.random.rand(784,784)
    currx = x[:, forInPutI]
    #deltaForWInPut = (getPhiDash(v) * delta).T.dot(wOutPut)
    phiDashOfV  = getPhiDash(v).reshape(784,1)
    deltaForWInPut = (wOutPut.T).dot(delta) * phiDashOfV
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
learningRate = 0.5
epoch = 0
while(not converged):
    epoch +=1
    erros = 0.0
    computedOP = []
    for index in range(0, trainingExamples):
        if(not isCorrectlyClassified(index)):
            erros+=1

    print("error % ", erros/trainingExamples)
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
    print("epoch ", epoch)
    print("winput ", wInPut[0][1:10])
    print("WOutPut ", wOutPut[0][1:10])