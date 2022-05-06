import pandas as pd
import numpy as np
from src.SimilarityArtworks import SimilarityArtworks

class ArtworksMatrix():
    def __init__(self, data_art, function_dict, weight_dict):#, colors_path=None):
        self.data_art = data_art
        self.function_dict = function_dict
        self.weight_dict = weight_dict
        # self.colors_path = colors_path
        self.simArtworks = SimilarityArtworks()
        
    def getSimilarityMatrix(self):
        artworks_matrix = np.zeros((len(self.data_art),len(self.data_art)))
        i = 0
        for a in self.data_art['ID']:
            j = 0
            for b in self.data_art['ID']:
                artworks_matrix[i][j] = self.computeSimilarity(a, b)
                j+=1
            i+=1
            
        return pd.DataFrame(artworks_matrix, index = [i for i in self.data_art['ID']], columns = [i for i in self.data_art['ID']])
        
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        total_sim = 0
        for col, fun in self.function_dict.items():
            if fun is not None:
                if col not in self.data_art.columns: # Atributo con fichero externo
                    result = getattr(self.simArtworks, fun)(self.data_art, col).computeSimilarity(A, B)
                else:
                    result = getattr(self.simArtworks, fun)(self.data_art).computeSimilarity(A, B)
                    
                total_sim += result * self.weight_dict.get(col)
        
        # if self.colors_path is not None:
        #     total_sim += self.simArtworks.SimilarityColors(self.data_art, self.colors_path).computeSimilarity(A, B) * self.weight_dict.get("Colors")
        
        
        return total_sim