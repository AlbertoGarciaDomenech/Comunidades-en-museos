import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN

class UsersClustering:    
    
    def __init__(self, data):
        self.data = data
        
    def daviesBouldinScore(self, min_clusters=2, max_clusters=11):
        """Compute Davies Bouldin score, and returns the number of clusters with the lowest score"""
        davies_bouldin = np.zeros(max_clusters-2)
        for k in range(2, max_clusters):
            km = KMedoids(metric='precomputed', n_clusters=k, init='k-medoids++')
            km.fit(self.data)
            davies_bouldin[k-2] = davies_bouldin_score(self.data, km.labels_)

        return range(2, max_clusters)[np.argmin(davies_bouldin)]
    
    def kMedoidsFromMatrix(self, n_clusters=None, max_clusters=11):
        """Recieves a similarity matrix, runs Kmedoids algorithm on those elemnts and returns the cluster asigned to each element"""
        if n_clusters is None:
            n_clusters = self.daviesBouldinScore(max_clusters=max_clusters)
        
        kmedoids = KMedoids(metric='precomputed',method='pam', n_clusters=n_clusters, init='k-medoids++')
        kmedoids.fit(self.data)
        
        #TODO: Tupla (id, label)
        return kmedoids.labels_
    
    def agglomerativeFromMatrix(self, n_clusters=None, max_clusters=11):
        """Recieves a similarity matrix, runs Kmedoids algorithm on those elemnts and returns the cluster asigned to each element"""
        if n_clusters is None:
            n_clusters = self.daviesBouldinScore(max_clusters=max_clusters)
        agg = AgglomerativeClustering(n_clusters = n_clusters, affinity='precomputed', distance_threshold = None, linkage = 'average')
        agg.fit(self.data)
        
        #TODO: Tupla (id, label)
        return agg.labels_

    def dbscanFromMatrix(self, n_clusters=None, max_clusters=11):
        """Recieves a similarity matrix, runs Kmedoids algorithm on those elemnts and returns the cluster asigned to each element"""
        # if n_clusters is None:
        #     n_clusters = self.daviesBouldinScore(max_clusters=max_clusters)
        dbscan = DBSCAN( metric='precomputed' , eps = .1, min_samples = 7)
        dbscan.fit(self.data)
        
        #TODO: Tupla (id, label)
        return dbscan.labels_