import pandas as pd
import numpy as np
import abc
import json
import colorsys
import math
from src.SimilarityFunctionInterface import *

# ARTWORKS_CSV = 'data/Prado_artworks_wikidata.csv'

class SimilarityArtworks():
    
    class SimilarityArtist(SimilarityFunctionInterface):
        """Compute similarity between artworks (by artist)"""
        def __init__(self, data_art):
            self.data = data_art

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFunctionInterface.computeSimilarity()"""
            return (self.data.loc[self.data['ID'] == A]['Artist'].to_list()[0] == self.data.loc[self.data['ID'] == B]['Artist'].to_list()[0])

    class SimilarityCategory(SimilarityFunctionInterface):
        """Compute similarity between artworks (by category)"""
        def __init__(self, data_art):
            self.data = data_art

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFunctionInterface.computeSimilarity()"""
            return(self.data.loc[self.data['ID'] == A]['Category'].to_list()[0] == self.data.loc[self.data['ID'] == B]['Category'].to_list()[0])

    class SimilarityColors(SimilarityFunctionInterface):
        """Compute similarity between artworks (by color)"""
        def __init__(self, data_art, colors_json):
            with open(colors_json)as f:
                self.colors = json.load(f)
                self.data = data_art

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

            dh = min(abs(hsv_a[0]-hsv_b[0]), 360-abs(hsv_a[0]-hsv_b[0])) / 180.0
            ds = abs(hsv_a[1] - hsv_b[1])
            dv = abs(hsv_a[2] - hsv_b[2]) / 255.
            distance = math.sqrt(dh * dh + ds * ds + dv * dv)
            return round(1. - (distance), 2)
        