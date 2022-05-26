import pandas as pd
import numpy as np
from src.SimilarityUsers import SimilarityUsers

class UsersMatrix():
    def __init__(self, data_users, artworks_sim, function_dict, weight_dict):
        self.data_users = data_users
        self.artworks_sim = artworks_sim
        self.function_dict = function_dict
        self.weight_dict = weight_dict
        self.simUsers = SimilarityUsers()

    def getSimilarityMatrix(self):
        users_matrix = np.zeros((len(self.data_users), len(self.data_users)))
        i = 0
        for a in self.data_users['userId']:
            j = 0
            for b in self.data_users['userId']:
                sim = self.computeSimilarity(a, b)
                if sim >= 0:
                    users_matrix[i][j] = sim
                else:
                    users_matrix[i][j] = 0
                j += 1
            i += 1

        return pd.DataFrame(users_matrix, index = [i for i in self.data_users['userId']], columns = [i for i in self.data_users['userId']])

    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        total_sim = 0
        for col, fun in self.function_dict.items():
            if fun is not None:
                if 'Polarity' in fun:
                    super_weight = self.weight_dict.get('polarity')
                else:
                    super_weight = self.weight_dict.get('demographic')
                    
                result = getattr(self.simUsers, fun)(self.data_users, self.artworks_sim).computeSimilarity(A, B)
                total_sim += result * self.weight_dict.get(col) * super_weight
                
        return total_sim