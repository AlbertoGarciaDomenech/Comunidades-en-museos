import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from setup import PATHS, PARAMS
from sklearn.cluster import dbscan
from collections import OrderedDict
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from users_similarity import JaccardUserSimilarity
from artwork_similarity import Partial_Similarities
from cluster_visualization import clusterInfographic
from cluster_caracterization import clustersInformation


# Arguments parsing
parser = argparse.ArgumentParser(description='Community detector using DBSCAN or k-medoids')
parser.add_argument('-method', '-mt', choices=['dbscan', 'kmedoids'], default='dbscan', help='Method to be used')
parser.add_argument('-desc', '-d',  nargs=1, type=str,  default='No description',  help='A brief description of the parameters to be used')
for sim in Partial_Similarities:
    parser.add_argument('-w' + sim.getFullName(), '-w' + sim.getShortName(),  nargs=1, type=int,  default=0,  help='Weight value for ' + sim.getFullName() + ' similarity')
parser.add_argument('-mode', '-md',  nargs=1, type=float,default=PARAMS['k_MOST_SIMILAR'],  help='Number of similar artworks to be retreived if mode [1, inf). If mode in [0, 1) mode value will be used as threshold to retreive those artworks with higher similarity')
parser.add_argument('-eps', nargs=1, type=float, default=PARAMS['DBSCAN_EPS'],  help='Epsilon value for DBSCAN clustering')
parser.add_argument('-minSamples', '-ms', nargs=1, type=int, default=PARAMS['DBSCAN_SAMPLES'],  help='Min samples in each cluster for DBSCAN clustering')
parser.add_argument('-minCenters', '-minK', nargs=1, type=int, default=PARAMS['KMEDOIDS_MIN_K'], help='Min clusters for KMEDOIDS')
parser.add_argument('-maxCenters', '-maxK', nargs=1, type=int, default=PARAMS['KMEDOIDS_MAX_K'], help='Max clusters for KMEDOIDS')
args = vars(parser.parse_args())

# Weight parser
weights = OrderedDict()
for sim in Partial_Similarities:
    if type(args[('w' + sim.getFullName())]) == list:
        weights[sim.getFullName().capitalize()] = args[('w' + sim.getFullName())][0]
    else:
        weights[sim.getFullName().capitalize()] = args[('w' + sim.getFullName())]

# Build the distance function 
sim = JaccardUserSimilarity(mode=args['mode'], weights=[*weights.values()])
def usersDistance(u1, u2):
    return 1 - sim.getSimilarity(u1[0], u2[0], polarity='positive')

# Run the clustering method

users = np.array(pd.read_csv(PATHS['USERS_DATA'])['userId']).reshape(-1,1)

params = {
    'weights'   : weights,
    'mode'      : args['mode']
}

if args['method'] == 'dbscan':
    params['eps'] = args['eps']
    params['min_samples'] = args['minSamples']
    labels = dbscan(X=users, eps=params['eps'], min_samples=params['min_samples'], metric=usersDistance, n_jobs=-1)[1]

if args['method'] == 'kmedoids':
    scores = []
    k_values = np.arange(args['maxCenters'] - args['minCenters'] + 1) + args['minCenters']
    for k_centers in k_values:
        km = KMedoids(n_clusters=k_centers, metric=usersDistance, random_state=PARAMS['RANDOM_STATE']).fit(users)
        score = silhouette_score(X=users, labels=km.labels_, metric=usersDistance)
        if score > max(scores, default = -1):
            labels = km.labels_
        scores.append(score)

metadata={'Method'  : args['method'].upper(),
          'Params'  : params,
          'Desc'    : args['desc'][0]}
info = clustersInformation(users.reshape(-1,), labels, metadata, "dbscan_" + str([*params['weights'].values()]))
clusterInfographic(info, args['method'] + "_" + str([*params['weights'].values()]))