import random
import pandas as pd
import numpy as np
import math
import operator
# Turn off warning pamda
pd.options.mode.chained_assignment = None  # default='warn'

# Load data and convert string to numeric values
def Data(file):
    data = pd.read_csv(file, names =['A','B','C','D','Species'])
    data.Species[data.Species == 'Iris-setosa'] = 0
    data.Species[data.Species == 'Iris-versicolor'] = 1
    data.Species[data.Species == 'Iris-virginica'] = 2
    return data

# Split data into trainingset and testset
def SplitData(data, split):
    trainingSet = []
    testSet = [] 
    for i in range(len(data)-1):
        if random.random() < split:
            trainingSet.append(data.values[i])
        else:
            testSet.append(data.values[i])
    return np.asarray(trainingSet), np.asarray(testSet)

# Calcualte distances L1 and L2 (Manhattan and Euclidean)
def Distance_L(Point1, Point2, L):
    if L == 0: # Euclidean
        distance = math.sqrt(np.sum(np.power(Point1[:len(Point1)-1] - Point2[:len(Point2)-1], 2)))
    elif L == 1: # Manhattan
        distance = np.sum(np.abs(Point1[:len(Point1)-1] - Point2[:len(Point2)-1]))
    return distance

# Find K neirnest neighbor
def Neighbors(trainingSet , testInstance, k, Distance):
    distances = []
    for i in range(len(trainingSet)):
        dist = Distance_L(testInstance , trainingSet[i], Distance)
        distances.append((trainingSet[i] , dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return np.asarray(neighbors)

# Find Majority
def Majority(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# Find accuracy
def Accuracy(testSet, predictions):
    correctPredictions = 0
    for i in range(len(testSet)):
        if testSet[i][-1] is predictions[i]:
            correctPredictions +=1
    return (correctPredictions/float(len(testSet))) * 100.0

# Main
def Main():
    data = Data('iris.csv')
    Split = 0.7
    trainingSet, testSet = SplitData(data, Split)
    print('Training set ' + repr(len(trainingSet)))
    print('Test set ' + repr(len(testSet)))
    predictions = []
    k = 3
    Distance = 0
    for i in range(len(testSet)):
        # Find k neighbors for the i-th testset
        neighbors = Neighbors(trainingSet , testSet[i], k, Distance)
        # Get the predictions
        result = Majority(neighbors)
        predictions.append(result)
        # testSet[x][-1] returns the last value of the array 
        # (e.g a = [1,2,3] a[-1] = 3) 
        print('predicted = ' + repr(result) + ', actual = ' + repr(testSet[i][-1]))
    accuracy = Accuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy)+'%')

Main()