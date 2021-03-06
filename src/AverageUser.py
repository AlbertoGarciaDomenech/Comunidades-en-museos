import pandas as pd
import numpy as np
import collections
import json
import matplotlib.pyplot as plt
import os

class AverageUser:
    
    def __init__(self, data, artworks_info, atributes_users, atributes_artworks):
        self.data = data
        self.artworks_info = artworks_info
        self.atributes_users = atributes_users
        self.atributes_artworks = atributes_artworks
        self.stats_dicts = {}
        self.image_path = "data/cache/"
    
    def computeAverageUser(self, n_artworks=3):
        self.n_artworks = n_artworks
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
                    
            for atr in ['positive', 'negative', 'mixed']:
                # for atr in self.atributes_artworks:
                artworks = collections.Counter([artw for usr in c[atr] for artw in usr])
                averageUser[atr] = [item[0] for item in artworks.most_common(n_artworks)]

            clusterId = c.cluster.iloc[0]
            self.stats_dicts[clusterId] = aux_dict
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
                print("\tIndividuos: ", len(self.data[self.data.cluster == cluster]))
                for atr, perc in d.items():
                    if atr not in self.atributes_users:
                        continue
                    print("\t" + atr,":")
                    for x in perc:
                        print("\t\t",  x[0], "(", x[1], "%)")
                
                for polarity in np.intersect1d(self.atributes_users,['positive', 'negative', 'mixed']):
                    print("\t--Top",self.n_artworks, " " + polarity + "--")
                    if(len(self.users_df[self.users_df['cluster'] == cluster][polarity]) == 0):
                        print("0 artworks found with", polarity, " polarity")
                    for artworks in self.users_df[self.users_df['cluster'] == cluster][polarity]:
                        for art in artworks:
                            print("\t\t\tTitle:" + self.artworks_info[self.artworks_info['ID'] == art]['Title'].to_list()[0])
                            for col in self.artworks_info.columns:
                                if col in self.atributes_artworks:
                                    print("\t\t\t" + col + ": " + self.artworks_info[self.artworks_info['ID'] == art][col].to_list()[0])
        
    def returnExplanation(self):
        explanation = {}
        if len(self.stats_dicts) != 0:
            for cluster, d in self.stats_dicts.items():
                explanation[cluster] = ("<p>Individuos: "+ str(len(self.data[self.data.cluster == cluster])) + "</p>")
                # print("\tIndividuos: ", len(self.data[self.data.cluster == cluster]))
                for atr, perc in d.items():
                    if atr not in self.atributes_users:
                        continue
                    explanation[cluster] += ("<p>" + str(atr) +":</p>")
                    for x in perc:
                        explanation[cluster] +=("<p>\t" +  str(x[0]) + "("+ str(x[1]) + "%)</p>")
                
                for polarity in np.intersect1d(self.atributes_users,['positive', 'negative', 'mixed']):
                    explanation[cluster] +=("<p>\t--Top" + str(self.n_artworks) + " " + str(polarity) + "--</p>")
                    if(len(self.users_df[self.users_df['cluster'] == cluster][polarity]) == 0):
                        explanation[cluster] += ("<p>0 artworks found with" + str(polarity) + " polarity</p>")
                    for artworks in self.users_df[self.users_df['cluster'] == cluster][polarity]:
                        for art in artworks:
                            explanation[cluster] += ("<p>\t\t\tTitle:" + str(self.artworks_info[self.artworks_info['ID'] == art]['Title'].to_list()[0]) + "</p>")
                            for col in self.artworks_info.columns:
                                if col in self.atributes_artworks:
                                    explanation[cluster] += ("<p>\t\t\t" + str(col) + ": " + str(self.artworks_info[self.artworks_info['ID'] == art][col].to_list()[0])  + "</p>")
        return explanation
    
    def returnJSONExplanation(self):
        explanation = {}
        if len(self.stats_dicts) != 0:
            for cluster, d in self.stats_dicts.items():
                explanation[int(cluster)] = {}
                explanation[int(cluster)]['usr'] = {}
                explanation[cluster]['usr']['Individuos'] = len(self.data[self.data.cluster == cluster])
                for atr, perc in d.items():
                    if atr not in self.atributes_users:
                        continue
                    explanation[int(cluster)]['usr'][atr] = {} 
                    for x in perc:
                        explanation[int(cluster)]['usr'][atr][str(x[0])] = x[1]
                explanation[int(cluster)]['polarity'] = {}
                for polarity in np.intersect1d(self.atributes_users,['positive', 'negative', 'mixed']):
                    explanation[int(cluster)]['polarity'][polarity] = [{} for i in range(len(self.users_df[self.users_df['cluster'] == cluster][polarity].to_list()[0]))] 
                    if not(len(self.users_df[self.users_df['cluster'] == cluster][polarity].to_list()[0]) == 0):
                        i = 0
                        for artworks in self.users_df[self.users_df['cluster'] == cluster][polarity].to_list()[0]:
                            explanation[int(cluster)]['polarity'][polarity][i]= {'title': str(self.artworks_info[self.artworks_info['ID'] == artworks]['Title'].to_list()[0])}
                            for col in self.artworks_info.columns:
                                if col in self.atributes_artworks:
                                     explanation[int(cluster)]['polarity'][polarity][i][col] = str(self.artworks_info[self.artworks_info['ID'] == artworks][col].to_list()[0])

                            i +=1
        return (json.dumps(explanation))
    
    def computeInfographics(self):
        dir = os.getcwd() + "/" + self.image_path
        for f in os.listdir(dir):
            if f != '.ipynb_checkpoints':
              os.remove(os.path.join(dir, f))
        
        if len(self.stats_dicts) != 0:
            for cluster, d in self.stats_dicts.items():
                info = self.data[self.data.cluster == cluster]
                for atr,perc in d.items():
                    if atr not in self.atributes_users:
                        continue
                    fig, ax = plt.subplots()
                    
                    names = info.groupby(atr).size().axes[0]
                    values = info.groupby(atr).size().to_list()
                    # bars = ax.bar(np.sort(names),np.sort(values))
                    bars = ax.bar(names,values)
                    ax.bar_label(bars, label_type = 'center')
                    plt.xticks(rotation=15)
                    # plt.bar(np.sort(names),np.sort(values), width = 0.5)
                    img_name = self.image_path + str(cluster) + "_" + str(atr) + ".png"
                    plt.savefig(img_name)
                    plt.clf()
                    plt.close()