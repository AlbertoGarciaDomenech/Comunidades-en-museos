import pandas as pd
import numpy as np
import collections

class AverageUser:
    
    def computeAverageUser(self, data, n_artworks=3):
        clusters = []
        for c in data.cluster.unique():
            clusters.append(data[data['cluster'] == c])
        
        newUsers = []
        for i, c in enumerate(clusters):
            averageUser = {'userId' : 0}
            for atr in c.columns:
                if atr not in ['userId', 'cluster', 'positive', 'negative', 'mixed']:
                    averageUser[atr] = c[atr].mode()[0] ###### NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO (pueden salir varios)         
                    
            for atr in c.columns:
                if atr in ['positive', 'negative', 'mixed']:
                    artworks = collections.Counter([artw for usr in c[atr] for artw in usr])
                    averageUser[atr] = [item[0] for item in artworks.most_common(n_artworks)]
                        
            averageUser['cluster'] = i
            newUsers.append(averageUser)
            
        return pd.DataFrame(newUsers)