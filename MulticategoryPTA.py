import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
import matplotlib.pyplot as plt

# Reading input from the images
np.set_printoptions(threshold=np.inf)
flbl = open("train-labels.idx1-ubyte", 'rb')
magic_nr, size_label = struct.unpack(">II", flbl.read(8))
lbl = pyarray("b", flbl.read())
flbl.close()

fimg = open("train-images.idx3-ubyte", 'rb')
magic_nr, size_img, rows, cols = struct.unpack(">IIII", fimg.read(16))
img = pyarray("B", fimg.read())
img = np.asarray(img).reshape(60000, 784)
fimg.close()

desired_label = np.zeros((size_img,10), dtype=np.int)
for i in range(size_label):
	desired_label[i][lbl[i]] = 1

# end of reading input

epoch = 0
n = 1
e = 0.1
errors = []
epochs = []
w = np.random.rand(10,784)
# x = np.random.rand(6000, 784,1)
# d = np.random.rand(6000,10,1)
# v = np.random.rand(10,1)
def MPTA():
    global epoch, w, img
    toContinue = True
    # getting the no of images
    #i = 5000
    i = img.shape[0]
    while(toContinue):
        print("epoch :: ", epoch)
        errors.append(0)
        for index in range(0, i):
            if index%10000 == 0:
                print(index, " elements done")
            x = img[index].reshape(784,1)
            d = desired_label[index].reshape(10,1)
            #errors[epoch]=0
            v = w.dot(x)

            maxIndex = findMaxIndex(v)
            v = getVectorWithOneAtIndex(maxIndex)
            if (not (d==v).all()):
                errors[epoch] = errors[epoch]+1

        epoch = epoch +1
        epochs.append(epoch)
        #print(errors)
        # updating weights

        #print("before updating weights")
        #print(w[0][:50])
        for index in range(0,i):
            x = img[index].reshape(784, 1)
            d = desired_label[index].reshape(10, 1)
            w = w + n * (d - applyStepFunctionOnV(w.dot(x))) * np.transpose(x)

        #print("after updating weights")
        #print(w[0][:50])

        # cehcking if the minimum error ratio is reached or not
        if (errors[epoch-1]/i < e):
            toContinue = False
            print("error rate :: ", errors[epoch - 1] / i)
        else:
            print("errors not meeting standards. error rate :: ", errors[epoch-1]/i)

    print(" optimal weights found after", epoch, " epochs")
    plt.plot(epochs, errors, 'b')
    plt.axis([0, epoch, 0, max(errors)])
    plt.show()

# Testing
def testImages():
    i = img.shape[0]
    misClassErrorsInTesting = 0
    for index in range(0, i):
        if index%10000 == 0:
            print(index, " elements done")
        x = img[index].reshape(784,1)
        d = desired_label[index].reshape(10,1)
        #errors[epoch]=0
        v = w.dot(x)

        maxIndex = findMaxIndex(v)
        v = getVectorWithOneAtIndex(maxIndex)
        if (not (d==v).all()):
            misClassErrorsInTesting = misClassErrorsInTesting + 1
    print (misClassErrorsInTesting)



# applying step function on W*V
def applyStepFunctionOnV(tempV):
    for j in range (0,10):
        tempV[j][0] = stepFunction(tempV[j][0])
    return tempV


def stepFunction(value):
    return 1 if value>=0 else 0


# this will return a matrix of size 10X1 with 1 at given index and 0 at other indices
# this is used to compare the computerd class to expected class Di
def getVectorWithOneAtIndex(maxIndex):
    tempV =np.random.rand(10,1)
    tempV.fill(0)
    tempV[maxIndex][0]=1
    return tempV

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



v = np.random.rand(10,1)
v[5][0] = -1
#print (v)
#print(applyStepFunctionOnV(v))
#print (findMaxIndex(v))
#print (getVectorWithOneAtIndex(3))

# Training the network
MPTA()

#



