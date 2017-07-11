import random
import numpy as np
import math

file1 = open("/Users/owner/Desktop/test_data.txt", "r")
points = np.zeros(101, dtype='float32, float32, int32')
m = 0
for each_line in file1:
    x = each_line.split()
    if m > 0:
        points[m][0] = float(x[1])
        points[m][1] = float(x[2])
    m = m+1


def kmeans(k):
    centroid = np.empty([k, 2], dtype='float32')
    for i in range(0, k):
        for j in range(0, 2):
            centroid[i][j] = random.random()
    return centroid


def classify(points, centroid):
    for i1 in range(1, len(points)):
        min1 = 10000.0
        temp = 0
        for j in range(0, len(centroid)):
            d = math.sqrt(math.pow(math.fabs(points[i1][0]-centroid[j][0]), 2)+math.pow(math.fabs(points[i1][1]-centroid[j][1]), 2))
            #print(d)
            if d < min1:
                min1 = d
                temp = j
        points[i1][2] = temp
    return points


def recomputeCentroid(points, centroid):
    points = sorted(points, key=lambda p: p[2])
    x = 0
    tempx = 0.0
    tempy = 0.0
    count = 0
    for index in range(1, len(points)):
        if x == points[index][2]:
            count = count + 1
            tempx = points[index][0] + tempx
            tempy = points[index][1] + tempy
        else:
            centroid[points[index-1][2]][0] = tempx/count
            centroid[points[index-1][2]][1] = tempy/count
            tempx = points[index][0]
            tempy = points[index][1]
            count = 1
    return centroid
def meanSquaredError(points,centroid):
    points = sorted(points, key=lambda p: p[2])
    tempx = 0.0
    tempy = 0.0
    count = 0
    d = 0.0
    for index in range(1, len(points)):
            d = d + math.sqrt(math.pow(math.fabs(points[index][0] - centroid[points[index-1][2]][0]), 2) + math.pow(
                math.fabs(points[index][1] - centroid[points[index-1][2]][1]), 2))
    return d


centroid = kmeans(5)
print("Initial Centroid")
print(centroid)
d = meanSquaredError(points, centroid)
print("Initial error")
print(d)
for i in range(0, 5):
    points = classify(points, centroid)
    d = meanSquaredError(points, centroid)
    centroid = recomputeCentroid(points, centroid)
print("Final Centroid")
print(centroid)
print("Mean Squared Error")
print(d)

