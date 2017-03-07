import itertools
import numpy as np
lst = list(itertools.product([-1, 1], repeat=3))
inputs = np.array(lst)
#w = np.array([[0, 2, -1], [2, 0, -1], [-1, -1, 1]])
w = np.array([[1, -1, -1], [-1, 0, 2], [-1, 2, 0] ])
b = np.array([0.5, 0.5, 0.5])
b = b.reshape(3,1)
biasTranspose = b.reshape(1,3)
for row in inputs:
    curip = row.reshape(3,1)
    #print (curip.reshape(1,3))
    res = (np.dot(w,curip) + b).reshape(1,3)
    curipTranspose = curip.reshape(1,3)

    e1 = -np.dot(curipTranspose, np.dot(w,curip)) - 2*np.dot(biasTranspose, curip)
    print(e1)
    #print(res)
    res =  [1 if(x>=0) else -1 for x in res[0]]
    #more testing
    #e2 = np.dot(curipTranspose, np.dot(w, curip)) - 2 * np.dot(biasTranspose, curip)
    print (curip.reshape(1,3), " -> ", res)
    #more testing
    print("***********")
    #testing git commit --amend
