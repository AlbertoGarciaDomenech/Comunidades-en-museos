import pandas as pd
import numpy as np
import collections

class AverageUser:
    
    def __init__(self, data):
        self.data = data
    
    def computeAverageUser(self, n_artworks=3):
        clusters = []
        for c in self.data.cluster.unique():
            clusters.append(self.data[self.data['cluster'] == c])
        
        newUsers = []
        newId = self.data['userId'][len(self.data['userId']) - 1] + 1
        for c in clusters:
            averageUser = {'userId' : 'expl'+str(newId)}
            for atr in c.columns:
                if atr not in ['userId', 'cluster', 'positive', 'negative', 'mixed']:
                    averageUser[atr] = c[atr].mode()[0] ###### NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO (pueden salir varios)         
                    
            for atr in c.columns:
                if atr in ['positive', 'negative', 'mixed']:
                    artworks = collections.Counter([artw for usr in c[atr] for artw in usr])
                    averageUser[atr] = [item[0] for item in artworks.most_common(n_artworks)]
                        
            averageUser['cluster'] = c.cluster.iloc[0]
            newUsers.append(averageUser)
            newId += 1
            
        return pd.DataFrame(newUsers)