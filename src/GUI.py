import ipywidgets as widgets
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt

from pyvis.network import Network # Visualización de los grafos
import networkx as nx
from ipywidgets import interact, interact_manual
import forceatlas2 as fa2
import random

import inspect

from src.SimilarityArtworks import *
from src.SimilarityUsers import *
from src.ArtworksMatrix import *
from src.UsersMatrix import *
from src.UsersClustering import *
from src.AverageUser import *
from src.UsersGraph import *


class GUI:
  
  def __init__(self, path='data/'):
    self.path = path
    self.simItemsPath = self.path + 'sim/sim items/'
    self.simUsersPath = self.path + 'sim/sim users/'
    self.files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
    self.simItemsFiles = [f for f in listdir(self.simItemsPath) if isfile(join(self.simItemsPath, f))]
    self.simUsersFiles = [f for f in listdir(self.simUsersPath) if isfile(join(self.simUsersPath, f))]
  
  def inner_classes_list(self, cls):
    return [cls_attribute.__name__ for cls_attribute in cls.__dict__.values() if inspect.isclass(cls_attribute)]
            # and issubclass(cls_attribute, SimilarityFunctionInterface)]

### 1. CARGAR DATOS ###
  def loadData(self):
    self.users = pd.read_csv(self.path + self.usersDropdown.value, index_col=0)
    self.items = pd.read_csv(self.path + self.itemsDropdown.value)
    self.users['positive'] = self.users['positive'].apply(eval)
    self.users['negative'] = self.users['negative'].apply(eval)
    self.users['mixed'] = self.users['mixed'].apply(eval)
    self.users_sim_functions = [None]
    self.users_sim_functions.extend(self.inner_classes_list(SimilarityUsers))
    self.items_sim_functions = [None]
    self.items_sim_functions.extend(self.inner_classes_list(SimilarityArtworks))

  def loadData_create_widgets(self):
    self.completeMessages = widgets.VBox([])
    self.loadDataComplete = widgets.Label(value="- Users and Items data loaded -")
    self.loadDataTitle = widgets.HTML(value="<h2>Select data files</h2>")
    self.usersDropdown = widgets.Dropdown(options=self.files,
                                          description='Users:',
                                          disabled=False,
                                          )
    self.itemsDropdown = widgets.Dropdown(options=self.files,
                                          description='Items:',
                                          disabled=False,
                                          )
    self.loadDataButton = widgets.Button(description="Done", button_style='success')     
    self.loadDataButton.on_click(self.on_loadDataButton_clicked)

  def on_loadDataButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      self.loadData()
      self.completeMessages.children += (self.loadDataComplete, )
      display(self.completeMessages)
      self.addItemsAtribute_create_widgets()
      display(self.addItemsAtributeTitle)
      display(self.addItemsAtributeButtonBox)

### 2. SIMILITUD ITEMS ###
  def computeItemsSim(self):
    self.items_funcs = {x : None for x in self.items.columns}
    self.items_weights = {x : None for x in self.items.columns}  

    for atr in self.items_atribute_list:
        if atr.children[0].value not in self.items.columns: # Atributo con fichero externo
            self.items_funcs.update({self.path + atr.children[0].value : atr.children[1].value})
            self.items_weights.update({self.path + atr.children[0].value : atr.children[2].value})
        else:
            self.items_funcs.update({atr.children[0].value : atr.children[1].value})
            self.items_weights.update({atr.children[0].value : atr.children[2].value})
    self.AM = ArtworksMatrix(self.items, function_dict=self.items_funcs, weight_dict=self.items_weights)#, colors_path=self.path + self.addItemsAtributeColor.children[0].value)
    self.itemsMatrix = self.AM.getSimilarityMatrix()

  def getItemsSim(self):
    self.items_funcs = {}
    self.itemsMatrix = pd.read_csv(self.simItemsPath + self.itemsSimilarityFileDropdown.value, index_col = 0)

  def addItemsAtribute_create_widgets(self):
    self.items_atribute_list = []
    self.default_items_funcs = []
    self.addItemsAtributeComplete = widgets.Label(value="- Items similarity computed -")
    self.addItemsAtributeTitle = widgets.HTML(value="<h2>Add atributes to compute ITEMS similarity</h2>")
    self.addItemsAtributeFileTitle = widgets.HTML(value="<h2>Select a file with the ITEMS similarity matrix</h2>")

    self.addItemsAtributeButton = widgets.Button(description="Atribute", icon='plus')
    self.addItemsAtributeButton.on_click(self.on_addItemsAtributeButton_clicked) 
    self.addItemsAtributeButtonExtra = widgets.Button(description="File Atribute", icon='plus')
    self.addItemsAtributeButtonExtra.on_click(self.on_addItemsAtributeButtonExtra_clicked)
    self.addItemsAtributeButtonFile = widgets.Button(description="Similarity File", icon='plus', button_style='info')
    self.addItemsAtributeButtonFile.on_click(self.on_addItemsAtributeButtonFile_clicked)        
    self.addItemsAtributeButtonBox = widgets.HBox([self.addItemsAtributeButton, self.addItemsAtributeButtonExtra,self.addItemsAtributeButtonFile])
    self.loadItemsAtributeButton = widgets.Button(description="Compute similarity", button_style='success')
    self.loadItemsAtributeButton.on_click(self.on_loadItemsAtributeButton_clicked)

    self.simItemsFileButton = widgets.Button(description="File selected", button_style='success',)
    self.simItemsFileButton.on_click(self.on_simItemsFileButton_clicked)

  def addItemsAtribute(self):
    aux1 = widgets.Dropdown(options=self.items.columns, value=None, description='Atribute:',disabled=False)
    aux2 = widgets.Dropdown(options=self.items_sim_functions, description='Sim Function:', disabled=False)
    # aux3 = widgets.BoundedFloatText(value=0.0, min=0, max=1, step=0.1, description='Weight:', disabled=False)
    aux3 = widgets.FloatSlider(value=0.0, min=0, max=1, step=0.1, description='Weight:', disabled=False, readout=True, readout_format='.1f')

    self.itemsLastAtribute = widgets.HBox([aux1, aux2, aux3])

  def addItemsAtributeExtra(self):
    aux1 = widgets.Dropdown(options=self.files, value=None, description='File:', disabled=False)
    aux2 = widgets.Dropdown(options=self.items_sim_functions, value=None, description='Sim Function:', disabled=False)
    # aux3 = widgets.BoundedFloatText(value=0.0, min=0, max=1, step=0.1, description='Weight:', disabled=False)
    aux3 = widgets.FloatSlider(value=0.0, min=0, max=1, step=0.1, description='Weight:', disabled=False, readout=True, readout_format='.1f')

    self.itemsLastAtribute = widgets.HBox([aux1, aux2, aux3])

  def addItemsSimilarityFile(self):
      self.itemsSimilarityFileDropdown = widgets.Dropdown(options=self.simItemsFiles, value=None, description='Sim file:', disabled=False)

  def on_addItemsAtributeButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      self.addItemsAtribute()
      display(self.completeMessages)
      display(self.addItemsAtributeTitle)
      for atr in self.items_atribute_list:
        display(atr)
      display(self.itemsLastAtribute)
      display(self.addItemsAtributeButtonBox)
      display(self.loadItemsAtributeButton)
      self.items_atribute_list.append(self.itemsLastAtribute)

  def on_addItemsAtributeButtonExtra_clicked(self, change):
    with self.out:
      self.out.clear_output()
      self.addItemsAtributeExtra()
      display(self.completeMessages)
      display(self.addItemsAtributeTitle)
      for atr in self.items_atribute_list:
        display(atr)
      display(self.itemsLastAtribute)
      display(self.addItemsAtributeButtonBox)
      display(self.loadItemsAtributeButton)
      self.items_atribute_list.append(self.itemsLastAtribute)    

  def on_addItemsAtributeButtonFile_clicked(self, change):
    with self.out:
      self.out.clear_output()
      self.addItemsSimilarityFile()
      display(self.completeMessages)
      display(self.addItemsAtributeFileTitle)
      display(self.itemsSimilarityFileDropdown)
      display(self.simItemsFileButton)
      # self.items_atribute_list.append(self.itemsLastAtribute)   

  def on_loadItemsAtributeButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      display(self.completeMessages)
      print('Computing items similarity...')
      self.computeItemsSim()
      self.out.clear_output()
      self.completeMessages.children += (self.addItemsAtributeComplete, )
      display(self.completeMessages)
      self.addUsersAtribute_create_widgets()
      display(self.addUsersAtributeTitle)
      # display(self.addUsersAtributeButton)
      display(self.addUsersAtributeButtonBox)    
  def on_simItemsFileButton_clicked(self, change):
      with self.out:
          self.out.clear_output()
          display(self.completeMessages)
          print('Getting items similarity file...')
          self.getItemsSim()
          self.out.clear_output()
          self.completeMessages.children += (self.addItemsAtributeComplete, )
          display(self.completeMessages)
          self.addUsersAtribute_create_widgets()
          display(self.addUsersAtributeTitle)
          for atr in self.users_atribute_list:
              display(atr)
          # display(self.addUsersAtributeButton)     
          display(self.addUsersAtributeButtonBox)


### 3. SIMILITUD USUARIOS ###
  def computeUsersSim(self):
    self.users_funcs = {x : None for x in self.users.columns}
    self.users_weights = {x : None for x in self.users.columns}  
    self.users_atribute_list.remove(self.demogPolBox)
    for atr in self.users_atribute_list:
      self.users_funcs.update({atr.children[0].value : atr.children[1].value})
      self.users_weights.update({atr.children[0].value : atr.children[2].value})

    self.users_weights.update({'polarity' : self.polWeight, 'demographic' : self.demogWeight}) ######

    self.UM = UsersMatrix(self.users, self.itemsMatrix, function_dict=self.users_funcs, weight_dict=self.users_weights)
    self.usersMatrix = self.UM.getSimilarityMatrix()

  def getUsersSim(self):
    self.users_funcs = {}
    self.usersMatrix = pd.read_csv(self.simUsersPath + self.usersSimilarityFileDropdown.value,index_col=0)

  def addUsersAtribute_create_widgets(self):
    self.loadUsersAtributeSlider = widgets.FloatSlider(value=0.0, min=0, max=1, step=0.1, disabled=False, readout=False)
    self.demogLabel = widgets.HTML(value='<font size=\"+0.5\">Demographic weight <b>1.0</b></font>')
    self.polLabel = widgets.HTML(value='<font size=\"+0.5\"><b>0.0</b> Polarity weight</b></font>')
    self.demogPolBox = widgets.HBox([self.demogLabel, self.loadUsersAtributeSlider, self.polLabel])
    self.demogWeight = 1.0
    self.polWeight = 0.0

    def link_sliders(change):
      self.demogWeight = 1 - change.new
      self.polWeight = change.new
      self.demogLabel.value = "<font size=\"+0.5\">Demographic weight <b>{:.1f}</b></font>".format(self.demogWeight)
      self.polLabel.value = "<font size=\"+0.5\"><b>{:.1f}</b> Polarity weight</font>".format(self.polWeight)

    self.loadUsersAtributeSlider.observe(link_sliders, names='value')

    self.users_atribute_list = [self.demogPolBox]
    self.default_users_funcs = []

    self.addUsersAtributeComplete = widgets.Label(value="- Users similarity computed -")
    self.addUsersAtributeTitle = widgets.HTML(value="<h2>Add atributes to compute USERS similarity</h2>")
    self.addUsersAtributeFileTitle = widgets.HTML(value="<h2>Select a file with the USERS similarity matrix</h2>")

    self.addUsersAtributeButton = widgets.Button(description="Atribute", icon='plus')
    self.addUsersAtributeButton.on_click(self.on_addUsersAtributeButton_clicked) 
    self.addUsersAtributeButtonFile = widgets.Button(description="Similarity File", icon='plus', button_style='info')
    self.addUsersAtributeButtonFile.on_click(self.on_addUsersAtributeButtonFile_clicked)        
    self.addUsersAtributeButtonBox = widgets.HBox([self.addUsersAtributeButton, self.addUsersAtributeButtonFile])

    self.loadUsersAtributeButton = widgets.Button(description="Compute similarity", button_style='success')
    self.loadUsersAtributeButton.on_click(self.on_loadUsersAtributeButton_clicked)

    self.loadUsersAtributeFileButton = widgets.Button(description="File selected", button_style='success',)
    self.loadUsersAtributeFileButton.on_click(self.on_simUsersFileButton_clicked)  

  def addUsersAtribute(self):
    aux1 = widgets.Dropdown(options=self.users.columns, value=None, description='Atribute:', disabled=False)
    aux2 = widgets.Dropdown(options=self.users_sim_functions, description='Sim Function::', disabled=False)
    # aux3 = widgets.BoundedFloatText(value=0.0, min=0, max=1, step=0.1, description='Weight:', disabled=False)
    aux3 = widgets.FloatSlider(value=0.0, min=0, max=1, step=0.1, description='Weight:', disabled=False, readout=True, readout_format='.1f')

    self.usersLastAtribute = widgets.HBox([aux1, aux2, aux3])

  def addUsersSimilarityFile(self):
      self.usersSimilarityFileDropdown = widgets.Dropdown(options=self.simUsersFiles, value=None, description='Sim file:', disabled=False)

  def on_addUsersAtributeButton_clicked(self, change):
    self.out.clear_output()
    self.addUsersAtribute()
    with self.out:
      display(self.completeMessages)
      display(self.addUsersAtributeTitle)
      for atr in self.users_atribute_list:
        display(atr)
      display(self.usersLastAtribute)
      # display(self.addUsersAtributeButton)
      display(self.addUsersAtributeButtonBox)
      display(self.loadUsersAtributeButton)
      self.users_atribute_list.append(self.usersLastAtribute)

  def on_addUsersAtributeButtonFile_clicked(self, change):
    with self.out:
      self.out.clear_output()
      self.addUsersSimilarityFile()
      display(self.completeMessages)
      display(self.addUsersAtributeFileTitle)
      display(self.usersSimilarityFileDropdown)
      display(self.loadUsersAtributeFileButton)
      # self.users_atribute_list.append(self.usersLastAtribute)

  def on_loadUsersAtributeButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      display(self.completeMessages)
      print('Computing users similarity...')
      self.computeUsersSim()
      self.out.clear_output()
      self.completeMessages.children += (self.addUsersAtributeComplete, )
      display(self.completeMessages)
      self.computeClusters_create_widgets()
      display(self.computeClustersTitle)
      display(self.clusterFunction)
      display(self.computeClustersButton)

  def on_simUsersFileButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      display(self.completeMessages)
      print('Getting users similarity file...')
      self.getUsersSim()

      self.out.clear_output()
      self.completeMessages.children += (self.addUsersAtributeComplete, )
      display(self.completeMessages)
      self.computeClusters_create_widgets()
      display(self.computeClustersTitle)
      display(self.clusterFunction)
      display(self.computeClustersButton)


### 4. CLUSTERING ###
  def computeClusters(self):
    users_distances = 1 - self.usersMatrix
    self.UC = UsersClustering(users_distances)
    funCluster = getattr(self.UC, self.clusterFunction.value)
    labels = funCluster()
    # labels = UsersClustering(users_distances).dbscanFromMatrix()
    self.users_clustered = self.users.copy()
    self.users_clustered['cluster'] = labels

  def computeClusters_create_widgets(self):
    self.computeClustersComplete = widgets.Label(value="- Users clustered -")
    self.computeClustersTitle = widgets.HTML(value="<h2>Compute users clusters</h2>")
    self.computeClustersButton = widgets.Button(description="Compute clusters", button_style='success')
    self.computeClustersButton.on_click(self.on_computeClustersButton_clicked)  
    self.clusterFunction = widgets.Dropdown(options=[x[0] for x in inspect.getmembers(UsersClustering, predicate=inspect.isfunction)[1:] if (x[0] != 'daviesBouldinScore')], value='kMedoidsFromMatrix', description='Alg Cluster:',disabled=False)

  def on_computeClustersButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      display(self.completeMessages)
      print('Computing users clusters...')
      self.computeClusters()
      self.out.clear_output()
      self.completeMessages.children += (self.computeClustersComplete, )
      display(self.completeMessages)
      self.showExplanation_create_widgets()
      display(self.showExplanationTitle)
      print("ITEM ATRIBUTES")
      display(self.addDefaultItemsCheckboxBox)
      print("USERS ATRIBUTES")
      display(self.addDefaultUsersCheckboxBox)
      display(self.showExplanationButton)

### 5. EXPLICACION ###
  def showExplanation(self):  
    self.users_atributes = self.default_users_funcs#[key for key, value in self.users_funcs.items() if value is not None] if len(self.default_users_funcs)==0 else self.default_users_funcs
    self.items_atributes = self.default_items_funcs#[key for key, value in self.items_funcs.items() if value is not None] if len(self.default_items_funcs)==0 else self.default_items_funcs
    self.AU = AverageUser(self.users_clustered, self.items, self.users_atributes, self.items_atributes)
    n_artworks = 3
    self.explicators = self.AU.computeAverageUser(n_artworks)
    self.AU.computeInfographics()
    self.expl = self.AU.returnExplanation()
    self.clustersTab.children = [widgets.HTML(value=str(self.expl[i]) +  "<img src=\" https://uploads7.wikiart.org/images/francisco-goya/charles-iv-of-spain-and-his-family-1800.jpg!Large.jpg\">") for i in self.expl.keys()]
    for i in range(len(self.clustersTab.children)):
      self.clustersTab.set_title(i,"cluster " + str(i))
    self.expl_json = self.AU.returnJSONExplanation()

  def explanation(self):
      path_img = self.path + "cache/"
      info = json.loads(self.expl_json)
      children_cluster = [widgets.Tab() for i in info]
      c = 0
      for cluster in info:
          # Demograficos
          demog_box = widgets.VBox()
          children_demog_box = [widgets.HBox() for atr in info[cluster]['usr']]
          i = 0
          for atr in info[cluster]['usr']:
              if atr == "Individuos":
                  demog_text_indiv = widgets.HTML(value = "<b>" + atr + "</b>: " + str(info[cluster]['usr'].get(atr)) + " de " + str(self.users.shape[0]) + "<br>")
                  children_demog_box[i] = demog_text_indiv
              else:
                  demog_box_atr = widgets.HBox()
                  demog_text_atr = widgets.HTML()
                  demog_text_atr.value += "<b>" + atr + "</b>:<br>"
                  for atr_mode in info[cluster]['usr'][atr]:
                      demog_text_atr.value += "&emsp;" + atr_mode + " ({:.2f}%)".format(info[cluster]['usr'][atr].get(atr_mode)) + "</br>"
                  img_name = path_img + str(cluster) + "_" + atr + ".png"
                  file = open(img_name, "rb")
                  image = file.read()
                  demog_img_atr =  widgets.Image(value = image, width=350, height=250)
                  infograph = widgets.Accordion(children = [demog_img_atr], selected_index=None)
                  infograph.set_title(0, "Info")
                  demog_box_atr.children = [demog_text_atr,infograph]
                  children_demog_box[i] = demog_box_atr

              i+=1
          demog_box.children = children_demog_box

          # Gustos
          children_pol = [widgets.Accordion(selected_index=None) for i in range(len(np.intersect1d(self.users_atributes,['positive', 'negative', 'mixed'])))]
          pol = 0
          for polarity in (np.intersect1d(self.users_atributes,['positive', 'negative', 'mixed'])):
              children_artworks = [widgets.HTML() for i in range(len(info[cluster]['polarity'][polarity]))]
              for artwork in range(len(info[cluster]['polarity'][polarity])):
                  for atr in info[cluster]['polarity'][polarity][artwork]:
                      if atr == "Image URL" :
                          children_artworks[artwork].value += "<p><img src=\"" + info[cluster]['polarity'][polarity][artwork].get(atr) + "\" width=\"325\"></p>" 
                      elif atr != "title" and atr != "Title":
                          children_artworks[artwork].value += "<p><b>" + atr + "</b>: " + info[cluster]['polarity'][polarity][artwork].get(atr) + "</p>"
              acc = widgets.Accordion(children = children_artworks, selected_index=None)
              for i in range(len(children_artworks)):
                  acc.set_title(i, info[cluster]['polarity'][polarity][i].get('title'))
              children_pol[pol] = acc
              pol += 1
          tabPol = widgets.Tab(children = children_pol)
          for i in range(len(children_pol)):
              tabPol.set_title(i, np.intersect1d(self.users_atributes,['positive', 'negative', 'mixed'])[i])

          # if the user only chooses demographic attributes to show
          if (len(info[cluster]["polarity"]) == 0):
              children_cluster[c] = demog_box
          else:
              children_cluster[c] = widgets.VBox([demog_box,tabPol])
          c +=1
      self.tabClusters = widgets.Tab(children = children_cluster)
      index = 0
      for j in info:
          self.tabClusters.set_title(index,"Cluster " + str(j))
          index+=1
      display(self.tabClusters)

  def showExplanation_create_widgets(self):
    self.clustersTab = widgets.Tab()
    self.showExplanationTitle = widgets.HTML(value="<h2>Clusters explanation</h2>")
    self.showExplanationButton = widgets.Button(description="Show explanation", button_style='success')
    self.showExplanationButton.on_click(self.on_showExplanationButton_clicked)      

    self.addDefaultItemsCheckbox = [widgets.Checkbox(description=atr, value=False, width='10px') for atr in self.items.columns]
    for i in self.addDefaultItemsCheckbox: 
      i.style.description_width = 'initial'
      if self.items_funcs.get(i.description) is not None: 
        i.value = True

    self.addDefaultItemsCheckboxBox = widgets.HBox(children = self.addDefaultItemsCheckbox, layout=widgets.Layout(width="800px"))

    self.addDefaultUsersCheckbox = [widgets.Checkbox(description=atr, width=1) for atr in self.users.columns]
    for i in self.addDefaultUsersCheckbox: 
      i.style.description_width = 'initial'
      if self.users_funcs.get(i.description) is not None: 
        i.value = True
    self.addDefaultUsersCheckboxBox = widgets.HBox(children = self.addDefaultUsersCheckbox, layout=widgets.Layout(width="800px"))

  def on_showExplanationButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      display(self.completeMessages)
      self.default_items_funcs = [x.description for x in self.addDefaultItemsCheckbox if x.value == True]
      self.default_users_funcs = [x.description for x in self.addDefaultUsersCheckbox if x.value == True]
      self.showExplanation()
      self.explanation()
      # display(self.clustersTab)
      self.showGraph_create_widgets()
      display(self.showGraphTitle)
      display(self.showGraphButton)

### 6. GRAFO ###
  def createGraph(self):
    self.UG = UsersGraph(artworks_data=self.items, users_funcs=self.users_funcs)

    # Añadir ususarios como nodos
    self.UG.addNodesFromUsers(self.users_clustered)
    # for i in range(len(self.users_clustered)): 
    #   usr = self.users_clustered.loc[i].copy()
    #   usr['positive'] = [self.items.loc[self.items['ID'] == id, 'Title'].item() for id in usr['positive']]
    #   usr['negative'] =  [self.items.loc[self.items['ID'] == id,'Title'].item() for id in usr['negative']]
    #   usr['mixed'] = [self.items.loc[self.items['ID'] == id,'Title'].item() for id in usr['mixed']]
    #   title = '<p>' + ''.join('{bold}'.format(bold = '<b>' if self.users_funcs.get(self.users_clustered.columns[e]) is not None else '')
    #                           + str(self.users_clustered.columns[e]) 
    #                           + '{endBold} :'.format(endBold = '</b>' if self.users_funcs.get(self.users_clustered.columns[e]) is not None else '') 
    #                           + str(usr[e]) 
    #                           + "<br>" for e in range(len(usr.to_list()))) +'</p>'
    #   self.G.add_nodes_from([(int(usr.userId), {'pos' : 0, 'color' : self.colors[usr.cluster], 'title' : title, 'label' : str(usr.userId)})])

    # Añadir aristas con similitud de usuarios
    self.UG.addEdgesFromSim(self.usersMatrix)
    # for i in range(len(self.usersMatrix.values)): 
    #   for j in range(len(self.usersMatrix.values[i])):
    #     if i != j and self.usersMatrix.values[i][j] > 0:
    #       a = list(self.G.nodes)[i]
    #       b = list(self.G.nodes)[j]
    #       self.G.add_edges_from([(a, b, {'weight' : self.usersMatrix.values[i][j]})])

    ## INDIVIDUOS EXPLICADORES (solo si no se ha usado matriz externa)      
    if hasattr(self, 'UM'):
      self.UG.addNodesFromUsers(self.explicators, shape='diamond')
      # for i in range(len(self.explicators)): # Añadir explicadores como nodos
        # usr = self.explicators.loc[i].copy()
        # usr['positive'] = [self.items.loc[self.items['ID'] == id, 'Title'].item() for id in usr['positive']]
        # usr['negative'] =  [self.items.loc[self.items['ID'] == id,'Title'].item() for id in usr['negative']]
        # usr['mixed'] = [self.items.loc[self.items['ID'] == id,'Title'].item() for id in usr['mixed']]
        # title = '<p>' + ''.join('{bold}'.format(bold = '<b>' if self.users_funcs.get(self.explicators.columns[e]) is not None else '')
        #                         + str(self.explicators.columns[e]) 
        #                         + '{endBold} :'.format(endBold = '</b>' if self.users_funcs.get(self.explicators.columns[e]) is not None else '') 
        #                         + str(usr[e]) 
        #                         + "<br>" for e in range(len(usr.to_list()))) +'</p>'
        # self.UG.G.add_nodes_from([(usr.userId, {'pos' : 0, 'shape' : 'diamond', 'color' : self.UG.colors[usr.cluster], 'title' : title, 'label' : str(usr.userId)})])

      del self.UM
      del Singleton._instances
      Singleton._instances = {}
      users_expl = pd.concat([self.users_clustered, self.explicators], ignore_index=True)
      self.UM = UsersMatrix(users_expl, self.itemsMatrix, function_dict=self.users_funcs, weight_dict=self.users_weights)

      # creamos matriz nueva de similitud incluyendo a los explicadores
      self.explMatrix = self.usersMatrix.copy()
      users_expl = pd.concat([self.users_clustered, self.explicators], ignore_index=True)
      # lista similitud entre explicadores y el resto de usuarios
      list_sim = [[] for i in range(len(self.explicators))]
      k = 0
      for i in self.explicators['userId']:
          for j in users_expl['userId']:
              list_sim[k].append((self.UM.computeSimilarity(i,j)))
          k +=1
      # creamos nuevas columnas para los explicadores
      for i in self.explicators['userId']:
          self.explMatrix[i] = 0
      # creamos nuevas filas con la similitud calculada antes
      df_new_row = pd.DataFrame(data=np.array([list_sim[k] for k in range(len(self.explicators))]), columns=self.explMatrix.columns)
      self.explMatrix = pd.concat([self.explMatrix,df_new_row], ignore_index=True)
      # cambiamos el valor de las columnas de los explicadores a sus similitudes
      index_expl = 0
      for i in self.explicators['userId']:
          self.explMatrix[i] = list_sim[index_expl]
          index_expl+=1
      self.explMatrix = self.explMatrix.set_axis(users_expl['userId'].to_list(), axis='index')

      self.UG.addEdgesFromSim(self.explMatrix)
      # for i in range(len(self.explicators)): # Añadir aristas de explicadores
      #   for j in range(len(self.users_clustered)):
      #   # if self.explicators.cluster[i] == self.users_clustered.cluster[j]:
      #     a = self.explicators.userId[i]
      #     b = list(self.UG.G.nodes)[j]
      #     weight = self.UM.computeSimilarity(a, self.users_clustered.userId[j])
      #     if weight > 0:
      #       self.UG.G.add_edges_from([(a, b, {'weight' : weight})])


    self.UG.applyForceAtlas2()
    # pos = { i : (random.random(), random.random()) for i in self.G.nodes()} # Optionally specify positions as a dictionary 
    # l = fa2.forceatlas2_networkx_layout(self.G, pos, niter=100, edgeWeightInfluence=3, scalingRatio=5.0, gravity=100)
    # x = { i : (random.random(), random.random()) for i in self.G.nodes()}
    # y = { i : (random.random(), random.random()) for i in self.G.nodes()}
    # for k, v in l.items():
    #   x.update({k : v[0]})
    #   y.update({k : v[1]})
    # nx.set_node_attributes(self.G, x , name='x')
    # nx.set_node_attributes(self.G, y , name='y')

    self.net = Network(notebook=True, width='90%')
    self.net.from_nx(self.UG.G)
    for edge in self.net.edges:
      edge.update({'hidden' : True})

    for node in self.net.nodes:
      node.pop('label', None)

  def showGraph(self):
    self.net.toggle_physics(False)
    # self.net.width = '75%'
    # self.net.show_buttons(filter_=['physics'])
    return self.net.show('nodes.html')

  def showGraph_create_widgets(self):
    self.showGraphTitle = widgets.HTML(value="<h2>Users graph</h2>")
    self.showGraphButton = widgets.Button(description="Show graph", button_style='success')
    self.showGraphButton.on_click(self.on_showGraphButton_clicked)  

  def on_showGraphButton_clicked(self, change):
    with self.out:
      self.out.clear_output()
      display(self.completeMessages)
      display(self.tabClusters)
      print("Generating graph...")
      self.createGraph()
      self.out.clear_output()
      display(self.completeMessages)
      display(self.tabClusters)
      display(self.showGraph())

### MOSTAR TODO ###
  def display_widgets(self):
      self.loadData_create_widgets()
      self.out = widgets.Output()  # this is the output widget in which the df is displayed

      display(self.out)
      with self.out:
        display(self.loadDataTitle)
        display(widgets.VBox(
                            [
                                self.itemsDropdown,
                                self.usersDropdown,
                                self.loadDataButton,
                            ]
                        )
               )