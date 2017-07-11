import json
import codecs
import re
from nltk.corpus import stopwords
from pandas import *
import numpy


regex_str = [
    r'(?:@[\w_]+)',  # @-mentions
    r'(?:[\w_]+)',  # other words
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    return filtered_words


def jaccardDist(setA, setB):
    return 1 - float(len(setA.intersection(setB))) / float(len(setA.union(setB)))


def finalProcessedData(location):
 data = []
 textList = []
 idList = []

 with codecs.open(location, 'rU', 'utf-8') as f:
     for line in f:
        data.append(json.loads(line))

 # Append empty lists in first two indexes.

 for i in data:
    str1 = preprocess(i['text'])
    textList.append(str1)
    idList.append(i['id'])

 finalData = [list(a) for a in zip(idList, textList)]
 return finalData


def kmeans(k, tweets, centroids):
    tweets = tweets.copy()
    tweets[2] = ""
    tweets[3] = ""
    for i in range(len(tweets[1])):
        set1 = set(tweets[1][i])
        jaccard = []
        for j in range(k):
            getIndex = Index(tweets[0]).get_loc(centroids[0][j])
            set2 = set(tweets[1][getIndex])
            dist = jaccardDist(set1, set2)
            jaccard.append(dist)
        tweets.set_value(i, 2, jaccard.index(min(jaccard)))
        tweets.set_value(i, 3, min(jaccard))
    tweets.columns = ['ID', 'Tweet', 'Cluster-ID','Jaccard-Dist']
    tweets = tweets.sort_values(['Cluster-ID', 'Jaccard-Dist'], ascending=[True, True])
    print(tweets['Tweet'])
    new_centroids = []
    for i in range(k):
        data_pts = tweets[tweets['Cluster-ID'] == i]
        if len(data_pts) != 0:
            mean_dist = numpy.mean(data_pts['Jaccard-Dist'])
            print("Data points for cluster:",i)
            print(data_pts['Jaccard-Dist'])
            print("Mean distance for cluster ", i, ":", mean_dist)
            print("Index closest to the distance:", numpy.argmin(numpy.abs(data_pts['Jaccard-Dist'] - mean_dist)))

            #Setting the new centroid from the data points as the one having distance closest to the mean of distance for that
            #cluster
            new_centroids.append(tweets['ID'][numpy.argmin(numpy.abs(data_pts['Jaccard-Dist'] - mean_dist))])
        else:
            new_centroids.append(centroids[0][i])
    for i in range(k):
        if new_centroids[i] != centroids[0][i]:
            centroids[0][i] = new_centroids[i]
            centroids[1][i] = 1
    return tweets


def sum_squared_errors(final_tweets):
    sse = 0
    for i in range(k):
        distance = 0
        data_pts = final_tweets[final_tweets['Cluster-ID'] == i]
        s1 = set(final_tweets['Tweet'][i])
        if len(data_pts) != 0:
            for j in range(len(data_pts)):
                s2 = set(final_tweets['Tweet'][data_pts.index.values[j]])
                jaccard_d = jaccardDist(s1, s2)
                distance += (jaccard_d)
            sse += distance
    return sse/10

#Main
k = 25
iterations = 5
finalData1 = finalProcessedData('C:/Users/indra/Desktop/Spring17/ML/Assignments/Assignment 6/Tweets.json')
tweets_original = DataFrame(finalData1)
tweets_final = []
centroids_final = []
#set initial seeds file location
centroids = read_csv('C:/Users/indra/Desktop/Spring17/ML/Assignments/Assignment 6/InitialSeeds.txt',
                     sep=",", header=None)


for i in range(iterations):
    tweets_final = kmeans(k, tweets_original, centroids)
    centroids_final = centroids

sum_sq_error = sum_squared_errors(tweets_final)

for i in range(len(centroids_final)):
    temp_list = tweets_final.loc[tweets_final['Cluster-ID'] == i]['ID']
    centroids_final[1][i] = temp_list.to_string(index=False).split('\n')

#set output location
outputfile = 'C:/Users/indra/Desktop/Spring17/ML/Assignments/Assignment 6/Output.txt'

centroids_final[1].to_csv(outputfile, sep=',', header=False)


with open(outputfile, 'a') as file:
    outputstr = "SSE: " + str(sum_sq_error)
    file.write(outputstr)




