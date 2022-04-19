import pandas as pd
import numpy as np
import collections

class AverageUser:
    
    def __init__(self, data, artworks_info, atributes_list):
        self.data = data
        self.artworks_info = artworks_info
        self.atributes_list = atributes_list
        self.stats_dicts = {}
    
    def computeAverageUser(self, n_artworks=3):
        clusters = []
        for c in self.data.cluster.unique():
            clusters.append(self.data[self.data['cluster'] == c])
        
        newUsers = []
        newId = self.data['userId'][len(self.data['userId']) - 1] + 1
        for c in clusters:
            averageUser = {'userId' : 'expl'+str(newId)}
            aux_dict = {}
            for atr in c.columns:
                if atr not in ['userId', 'cluster', 'positive', 'negative', 'mixed']:
                    mode = c[atr].mode()
                    perc = np.multiply(np.divide(c[atr].value_counts().to_list(), len(c)), 100)
                    
                    averageUser[atr] = mode[0]
                    
                    aux_dict[atr] = list(zip(mode, perc))
                    
            self.stats_dicts[c.cluster.iloc[0]] = aux_dict
                    
            for atr in c.columns:
                if atr in ['positive', 'negative', 'mixed']:
                    artworks = collections.Counter([artw for usr in c[atr] for artw in usr])
                    averageUser[atr] = [item[0] for item in artworks.most_common(n_artworks)]
                        
            averageUser['cluster'] = c.cluster.iloc[0]
            newUsers.append(averageUser)
            newId += 1
        
        self.users_df = pd.DataFrame(newUsers)
        return self.users_df
    
    def printExplanation(self):
        if len(self.stats_dicts) != 0:
            for cluster, d in self.stats_dicts.items():
                print('------------------------')
                print("Cluster: ", cluster)
                for atr, perc in d.items():
                    print(atr, perc)
                
                for polarity in ['positive', 'negative', 'mixed']:
                    print("\t" + polarity)
                    for artworks in self.users_df[self.users_df['cluster'] == cluster][polarity]:
                        for art in artworks:
                            for col in self.artworks_info.columns:
                                if col in self.atributes_list:
                                    print(col + ": " + self.artworks_info[self.artworks_info['ID'] == art][col].to_list()[0])
                    