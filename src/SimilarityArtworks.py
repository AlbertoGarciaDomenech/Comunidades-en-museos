import pandas as pd
import numpy as np
import abc
import json
import colorsys
import math
from src.SimilarityFunctionInterface import *

ARTWORKS_CSV = 'data/Prado_artworks_wikidata.csv'

class SimilarityArtist(SimilarityFunctionInterface):
    """Compute similarity between artworks (by artist)"""
    def __init__(self, data_csv=ARTWORKS_CSV):
        self.data = pd.read_csv(data_csv)
    
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        return (self.data.loc[self.data['ID'] == A]['Artist'].to_list()[0] == self.data.loc[self.data['ID'] == B]['Artist'].to_list()[0])
        
class SimilarityCategory(SimilarityFunctionInterface):
    """Compute similarity between artworks (by category)"""
    def __init__(self, data_csv=ARTWORKS_CSV):
        self.data = pd.read_csv(data_csv)

    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        return(self.data.loc[self.data['ID'] == A]['Category'].to_list()[0] == self.data.loc[self.data['ID'] == B]['Category'].to_list()[0])

class SimilarityColors(SimilarityFunctionInterface):
    """Compute similarity between artworks (by color)"""
    def __init__(self, data_csv=ARTWORKS_CSV, colors_json='data/artworkColors.json'):
        with open(colors_json)as f:
            self.colors = json.load(f)
            self.data = pd.read_csv(data_csv) 
        
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        A = list(self.data[self.data['ID'] == A]['wd:paintingID'])[0]
        B = list(self.data[self.data['ID'] == B]['wd:paintingID'])[0]
        max_a = int(max(self.colors.get(A).get('frequency'), key=self.colors.get(A).get('frequency').get))
        max_b = int(max(self.colors.get(B).get('frequency'), key=self.colors.get(B).get('frequency').get))
        
        rgb_a = self.colors.get(A).get('colors')[max_a]
        rgb_b = self.colors.get(B).get('colors')[max_b]
        
        hsv_a = colorsys.rgb_to_hsv(rgb_a[0], rgb_a[1], rgb_a[2])
        hsv_b = colorsys.rgb_to_hsv(rgb_b[0], rgb_b[1], rgb_b[2])
        
        ### Sacado de año anterior ###
        dh = min(abs(hsv_a[0]-hsv_b[0]), 360-abs(hsv_a[0]-hsv_b[0])) / 180.0
        ds = abs(hsv_a[1] - hsv_b[1])
        dv = abs(hsv_a[2] - hsv_b[2]) / 255.
        distance = math.sqrt(dh * dh + ds * ds + dv * dv)
        return round(1. - (distance), 2)
    
######################################################    
class SimilarityArtworks(SimilarityFunctionInterface):
    """Compute similarity between artworks"""
    def __init__(self, artist_weight=0.3, color_weight=0.3, category_weight = 0.4):
        self.artist_weight = artist_weight
        self.color_weight = color_weight
        self.category_weight = category_weight
        
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        artist_sim = SimilarityArtist().computeSimilarity(A, B)
        color_sim = SimilarityColors().computeSimilarity(A, B)
        cat_sim = SimilarityCategory().computeSimilarity(A,B)
        
        return (self.artist_weight * artist_sim) + (self.color_weight * color_sim)  + (self.category_weight * cat_sim)