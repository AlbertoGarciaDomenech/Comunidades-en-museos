import json
import numpy as np
import pandas as pd
from setup import PATHS
from collections import Counter
from users_similarity import myFavouriteArtworks

users = pd.read_csv(PATHS['USERS_DATA'])
artworks = pd.read_csv(PATHS['ARTWORKS_DATA'])
emotions = pd.read_csv(PATHS['EMOTIONS_DATA'])

users.replace('<12', '0-12', inplace=True)
users.replace('>70', '70+', inplace=True)

def separateClusters(objects, labels):
    clusters = dict()
    for c in np.unique(labels):
        pos = np.where(labels == c)
        if len(pos):
            clusters[c] = objects[pos]
    return clusters

def buildDistanceMatrix(distFun, userList):
    N = len(userList)
    distMatrix = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            dist = distFun(userList[i], userList[j])
            distMatrix[i][j] = dist
            distMatrix[j][i] = dist
    return distMatrix

def relabelClusters(labels):
    uniques = np.unique(labels)
    labelMap = dict(zip(uniques, np.arange(len(uniques))))
    mapper = np.vectorize(lambda l : labelMap[l])
    return mapper(labels.copy())

def getDemographicInformation(userIndices):
    information = dict()
    for column in ['country', 'age', 'gender']:
        information[column] = dict(users.iloc[userIndices][column].value_counts(normalize=True))
    return information

def getArtisticInformation(userIndices):
    information = dict()

    # Creamos un dataframe con los datos de las obras y la valoración neta de los usuarios del cluster
    clusterUsers = users.iloc[userIndices]['userId']
    positiveArtworks = dict(Counter(sum([list(myFavouriteArtworks(userId=u, polarity='positive')) for u in clusterUsers], [])))
    negativeArtworks = dict(Counter(sum([list(myFavouriteArtworks(userId=u, polarity='negative')) for u in clusterUsers], [])))
    net_valoration = { id : positiveArtworks.get(id,0) - negativeArtworks.get(id,0) for id in [*positiveArtworks] + [*negativeArtworks]}
    clusterArtworks = artworks.loc[artworks['ID'].isin([*net_valoration])][['ID', 'Title', 'Artist', 'Category', 'Image URL']]
    clusterArtworks['net_val'] = clusterArtworks['ID'].map(net_valoration)
    clusterArtworks.sort_values('net_val', ascending=False, inplace=True)

    # Sacamos los tops
    information['Most valued'] = [{'Title'      : aw['Title'],
                                   'Image'      : aw['Image URL'],
                                   'Positive'   : positiveArtworks.get(aw['ID'], 0),
                                   'Negative'   : negativeArtworks.get(aw['ID'], 0)} 
                                   for _, aw in clusterArtworks[0:3].iterrows()
                                   if net_valoration[aw['ID']] > 0]

    information['Least valued'] = [{'Title'      : aw['Title'],
                                    'Image'      : aw['Image URL'],
                                    'Positive'   : positiveArtworks.get(aw['ID'], 0),
                                    'Negative'   : negativeArtworks.get(aw['ID'], 0)} 
                                    for _, aw in clusterArtworks[-3:].iterrows()
                                    if net_valoration[aw['ID']] < 0]
                                    
    information['Popular artists'] = clusterArtworks.groupby('Artist').sum().sort_values('net_val', ascending=False)[0:3].index.to_list()
    information['Unpopular artists'] = clusterArtworks.groupby('Artist').sum().sort_values('net_val', ascending=False)[-3:].index.to_list()
    information['Top Category'] = clusterArtworks.groupby('Category').sum().sort_values('net_val', ascending=False)[0:1].index.to_list()[0]
    return information

def getEmotionsInformation(userIndices):
    return dict(emotions.loc[emotions['userId'].isin(users.iloc[userIndices]['userId'])]['emotion'].value_counts(normalize=True))

def clusterInformation(userIndices):
    information = dict()
    information['Size'] = len(userIndices)
    information['Demographics'] = getDemographicInformation(userIndices)
    information['Artistic'] = getArtisticInformation(userIndices)
    information['Emotions'] = getEmotionsInformation(userIndices)
    return information
    
def clustersInformation(objects, labels, metadata, dump=None):

    labels = relabelClusters(labels)

    info = {'Clusters' : {('Cluster ' + str(cluster)) : clusterInformation(users.loc[users['userId'].isin(clusterUsers)].index)
            for cluster, clusterUsers in separateClusters(objects, labels).items() if cluster > -1}}
    info['Metadata'] = metadata

    if dump is not None and type(dump) is str:
        with open(PATHS['CLUSTERS_JSON'] + dump.replace(' ','') + '.json', "w") as outfile: 
            json.dump(info, outfile, indent=4, sort_keys=True,
                    separators=(', ', ': '), ensure_ascii=False)

    return info
