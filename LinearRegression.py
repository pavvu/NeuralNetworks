import random
import numpy as np
import matplotlib.pyplot as plt

# code for finding optimal weights and computing cost is modfication of snippents from https://gist.github.com/marcelcaraciolo/1321575

# computing the cost at a given point
def compute_cost(X, actualY, weights):
    #Number of training samples
    m = len(actualY)
    computedY = X.dot(weights).flatten()
    squareOfErrors = (computedY - actualY) ** 2
    return (1.0 / (2 * m)) * squareOfErrors.sum()


# finding optimal weights
def findOptimalWeights(X, y, weights, learningRate, number_of_iterations):
    m = len(y)
    errorHistory = []
    THRESHOLD = 0.0001

    for index in range(number_of_iterations):
        computedY = X.dot(weights).flatten() # x.w for output

        # calculating erros
        errors_x1 = (computedY - y) * X[:, 0]
        errors_x2 = (computedY - y) * X[:, 1]

        #updating weights by taking the average of the errors
        weights[0][0] = weights[0][0] - learningRate *  (1.0 / m) * errors_x1.sum()
        weights[1][0] = weights[1][0] - learningRate * (1.0 / m) * errors_x2.sum()

        errorHistory.append(compute_cost(X, y, weights))

        # break if the error rate is not decreasing by a threshold
        if(index > 0 and abs(errorHistory[index-1] - errorHistory[index]) < THRESHOLD):
            break

    return weights

def generateXY():
    x = list(range(1,51))
    y = []
    for item in x:
        u = random.uniform(-1, 1)
        y.append(item + u)
    return x, y

def plotLineAndPoints(weights, x, y):
    # Plotting the line to be displayed
    x11 = ((weights[0] + weights[1] * 1))
    x12 = ((weights[0] + weights[1] * 50))
    plt.axis([0, 50, 0, max(y)])
    plt.plot([x11, x12], [1, 50], 'r')
    plt.scatter(x, y, marker='*', c='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def findLine():
    no_of_iterations = 40
    learningRate = 0.001
    x, y = generateXY()
    input = np.ones(shape=(50, 2))
    input[:, 1] = x
    #Initialize weights
    weights = np.zeros(shape=(2, 1))
    #find optimal weights
    weights = findOptimalWeights(input, y, weights, learningRate, no_of_iterations)
    print(weights)
    plotLineAndPoints(weights, x, y)


findLine()