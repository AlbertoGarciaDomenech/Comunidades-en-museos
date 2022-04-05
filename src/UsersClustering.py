import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score

class UsersClustering:    
        
    def daviesBouldinScore(self, data, min_clusters=2, max_clusters=11):
        davies_bouldin = np.zeros(max_clusters-2)
        for k in range(2, max_clusters):
            km = KMedoids(metric='precomputed', n_clusters=k)
            km.fit(data)
            davies_bouldin[k-2] = davies_bouldin_score(data, km.labels_)

        return range(2, max_clusters)[np.argmin(davies_bouldin)]
    
    def kMedoidsFromMatrix(self, data, n_clusters=None, max_clusters=11):
        if n_clusters is None:
            n_clusters = self.daviesBouldinScore(data=data, max_clusters=max_clusters)
        
        kmedoids = KMedoids(metric='precomputed', n_clusters=n_clusters)
        kmedoids.fit(data)
        
        #TODO: Tupla (id, label)
        return kmedoids.labels_