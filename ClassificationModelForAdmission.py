# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 23:38:59 2018

"""

from matplotlib import pyplot
import numpy as np
import pandas as pd

noOfFeatures = 2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def computeCost(X, y, thetas):  
    thetas = np.matrix(thetas)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * thetas.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * thetas.T)))
    return np.sum(first - second) / (len(X))

def calculateCostDerivatives(X, y, thetas):  
    thetas = np.matrix(thetas)
    X = np.matrix(X)
    y = np.matrix(y)
    costDerivatives = np.zeros(noOfFeatures+1)

    error = sigmoid(X * thetas.T) - y

    for i in range(noOfFeatures+1):
        term = np.multiply(error, X[:,i])
        costDerivatives[i] = np.sum(term) / len(X)

    return costDerivatives

def gradientDescent(X, y, thetas):
    alpha = 0.01
    tempThetas = np.zeros(noOfFeatures+1)
    isConverged = False
    previousCost = float('Inf')
    
    while not isConverged:
        costDerivatives = calculateCostDerivatives(X, y, thetas)
        for j in range(noOfFeatures+1):
            tempThetas[j] = thetas[j] - alpha * costDerivatives[j]
       
        thetas = tempThetas[:]
        currentCost = computeCost(X, y, thetas)
        diffInError = previousCost - currentCost
        if diffInError < 0:
            diffInError = -1*diffInError 
        if  diffInError < 1/1000:
            isConverged = True
        previousCost = currentCost
        print(currentCost)
    return thetas

def predict(thetas, ex):  
    thetas = np.matrix(thetas)
    ex = np.matrix(ex)
    probability = sigmoid(ex * thetas.T)
    if probability >= 0.5:
        return 1
    else:
        return 0
    #return [1 if x >= 0.5 else 0 for x in probability]

x1 = []
x2 = []
y = []
with open("ex2data1.txt") as f:
    for line in f:
        line = line.split(',')
        x1.append(float(line[0]))
        x2.append(float(line[1]))
        y.append(float(line[2]))
        
matrix = pd.DataFrame()

noOfExamples = len(x1)

matrix.insert(0, 'x0', np.ones(noOfExamples))
matrix.insert(1, 'x1', np.asarray(x1))
matrix.insert(2, 'x2', np.asarray(x2))
matrix.insert(3, 'y', np.asarray(y))

admitted = matrix[matrix.y == 1]
notAdmitted = matrix[matrix.y == 0]
pyplot.xlabel('Exam 1 score')        
pyplot.ylabel('Exam 2 score')

pyplot.plot(admitted.x1, admitted.x2, 'bp')
pyplot.plot(notAdmitted.x1, notAdmitted.x2, 'yo')

X = matrix[['x0', 'x1', 'x2']]
y = matrix[['y']]

X = np.array(X.values)  
y = np.array(y.values)  
thetas = np.zeros(noOfFeatures+1)

print("Cost before Applying Gradient Descent: ",computeCost(X, y, thetas))
thetas = gradientDescent(X, y, thetas)
print("Cost After Applying Gradient Descent: ",computeCost(X, y, thetas))
print(thetas)

#Drawing Decision Boundary
x1 = np.linspace(30, 100)
x2 = (-thetas[0] - thetas[1]*x1)/thetas[2]
pyplot.plot(x1, x2, 'k-')

#Evaluating Gradient Descent 
example = np.array([1, 45, 85])
print("Prediction for example: ", predict(thetas, example))
