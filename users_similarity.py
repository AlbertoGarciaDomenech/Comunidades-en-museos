import operator
import numpy as np
import pandas as pd
from setup import PATHS, PARAMS
from artwork_similarity import *

_emotionsData = pd.read_csv(PATHS['EMOTIONS_DATA'])
_artWorksData = pd.read_csv(PATHS['ARTWORKS_DATA'])

def myFavouriteArtworks(userId, polarity):
    """Devuelve el conjunto de todos los cuadros que tienen una polaridad positiva para una persona"""
    return np.unique(np.array(_emotionsData[(_emotionsData['userId'] == userId) & (_emotionsData['Polarity'] == polarity)]['artworkId']))

def idArtWorkToWD(idArtwork):
    """Dado un id de un cuadro devuelve su qualificador wd""" 
    return np.array(_artWorksData[_artWorksData['ID'] == idArtwork]['wd:paintingID'])[0]

def wdArtWorkToId(wdArtwork):
    """Dado un wd de un cuadro devuelve su identificador id"""
    return np.array(_artWorksData[_artWorksData['wd:paintingID'] == wdArtwork]['ID'])

class JaccardUserSimilarity(object):

    def __init__(self, mode=PARAMS['SIM_THRESHOLD'], weights=[]):
        """
        Jaccard User Similarity 

        Parameters:
        mode : int o double. Si es mayor que 1 se recuperaran los mode cuadros mas similares para cada ejemplo, si es menor se utilizará como umbral para recuperar los cuadros que esten por encima
        weights : vector de pesos para las similitudes parciales de cuadros
        polaridad : 
        """
        if mode < 0.:
            raise ValueError("Mode must be a number greater than one")
        # Compute most similar
        self.__mostSimilarArtworks__ = dict()
        for wd in _artWorksData['wd:paintingID']:
            if mode < 1:
                sim_set = mostSimilarArtworks(wd, threshold=abs(mode), weights=weights)
            else:
                sim_set = kMostSimilarArtworks(wd, k=round(mode), weights=weights)
            self.__mostSimilarArtworks__[wd] = set(map(lambda a : a[1], sim_set))

    def getSimilarity(self, user1, user2, polarity=PARAMS['DEFAULT_POLARITY']):
        """
        Devuelve el indice de similitud jaccard entre los conjuntos (dadas dos personas) de cuadros que les gustan y sus similares
        Parameters:
        user1 : IDs de los usuarios
        user2 : IDs de los usuarios
        polarity : polaridad de los cuadros a recuperar de cada usuario
        Returns:
        sim: coeficiente de similitud
        """
        artworks1 = set()
        for a1 in myFavouriteArtworks(user1, polarity):
            artworks1 |= self.__mostSimilarArtworks__[idArtWorkToWD(a1)]
        artworks2 = set()
        for a2 in myFavouriteArtworks(user2, polarity):
            artworks2 |= self.__mostSimilarArtworks__[idArtWorkToWD(a2)]
        
        if len(artworks1 | artworks2): # Si la union no es vacía devolvemos el coeficiente de Jaccard de ambos conjuntos
            return len(artworks1 & artworks2) / len(artworks1 | artworks2)
        else: # Si es vacía ...
            if user1 == user2: # ... y el usuario coincide, similitud máxima
                return 1.
            else: # ... y no coincide el usuario, similitud mínima
                return 0.


################################################################################
############################ Average 1v1 Similarity ############################
################################################################################

def averagePairSimilarity(userId1, userId2, polarity='positive', weights=[]):
    """
    Devuelve la similitud media de los cuadros que le gustan a ambos usuarios enfrentados uno a uno
    
    PARÁMETROS
        userId1, userId2    IDs de los usuarios
        polarity            polaridad de los cuadros con que se mide la similitud
    DEVUELVE
    """
    aw1 = myFavouriteArtworks(userId1, polarity)
    aw2 = myFavouriteArtworks(userId2, polarity)
    similarities = []
    for a1 in aw1: 
        for a2 in aw2:
            similarities.append(ArtworkSimilarity(idArtWorkToWD(a1), idArtWorkToWD(a2), weights))
    return np.average(np.array(similarities))


