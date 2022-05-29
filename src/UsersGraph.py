import pandas as pd
import networkx as nx
import forceatlas2 as fa2
import random

class UsersGraph:
    
  def __init__(self, artworks_data, users_funcs):
    self.colors = ["blue","orange","green","red","purple","brown","pink","gray", "olive", "cyan", "black", "b", "g", "r", "c", "m","y"]
    self.artworks_data = artworks_data
    self.users_funcs = users_funcs
    self.G = nx.Graph()

  def addNodesFromUsers(self, data, shape='dot'):
    for i in range(len(data)): 
      usr = data.loc[i].copy()
      usr['positive'] = [self.artworks_data.loc[self.artworks_data['ID'] == id, 'Title'].item() for id in usr['positive']]
      usr['negative'] =  [self.artworks_data.loc[self.artworks_data['ID'] == id,'Title'].item() for id in usr['negative']]
      usr['mixed'] = [self.artworks_data.loc[self.artworks_data['ID'] == id,'Title'].item() for id in usr['mixed']]
      title = '<p>' + ''.join('{bold}'.format(bold = '<b>' if self.users_funcs.get(data.columns[e]) is not None else '')
                              + str(data.columns[e]) 
                              + '{endBold} :'.format(endBold = '</b>' if self.users_funcs.get(data.columns[e]) is not None else '') 
                              + str(usr[e]) 
                              + "<br>" for e in range(len(usr.to_list()))) +'</p>'
      self.G.add_nodes_from([(str(usr.userId), {'pos' : 0, 'color' : self.colors[usr.cluster], 'shape' : shape, 'title' : title, 'label' : str(usr.userId)})])

  def addEdgesFromSim(self, sim_matrix):
      for i in range(len(sim_matrix.values)): 
        for j in range(len(sim_matrix.values[i])):
          if i != j and sim_matrix.values[i][j] > 0:
            a = list(self.G.nodes)[i]
            b = list(self.G.nodes)[j]
            self.G.add_edges_from([(a, b, {'weight' : sim_matrix.values[i][j]})])
              
  def applyForceAtlas2(self, _niter=100, _edgeWeightInfluence=3, _scalingRatio=5.0, _gravity=100):
    pos = { i : (random.random(), random.random()) for i in self.G.nodes()} # Optionally specify positions as a dictionary 
    l = fa2.forceatlas2_networkx_layout(self.G, pos, niter=_niter, edgeWeightInfluence=_edgeWeightInfluence, scalingRatio=_scalingRatio, gravity=_gravity)
    x = { i : (random.random(), random.random()) for i in self.G.nodes()}
    y = { i : (random.random(), random.random()) for i in self.G.nodes()}
    for k, v in l.items():
      x.update({k : v[0]})
      y.update({k : v[1]})
    nx.set_node_attributes(self.G, x , name='x')
    nx.set_node_attributes(self.G, y , name='y')