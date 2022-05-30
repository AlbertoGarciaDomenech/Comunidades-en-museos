import pandas as pd
import numpy as np
import math
from src.SimilarityFunctionInterface import *

COUNTRIES_CSV = "data/countries_of_the_world.csv"

class SimilarityUsers():

    class SimilarityAgeInterval(SimilarityFunctionInterface):
        """Compute similarity between users (by age)"""
        def __init__(self, data_users, artworks_sim):
            self.data = data_users
            self.age_index = 1
            self.preprocess()

        def preprocess(self):
            for age_range in self.data.sort_values(by=['age'])['age'].unique():
                self.data = self.data.replace(age_range, self.age_index)
                self.age_index += 1
            self.age_index -=1

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFuntionInterface.computeSimilarity()"""
            ageA = self.data.loc[self.data['userId'] == A]['age'].to_list()[0]
            ageB = self.data.loc[self.data['userId'] == B]['age'].to_list()[0]
            return 1 - (abs(ageA - ageB) / (self.age_index - 1))
        
    class SimilarityAgeNotInterval(SimilarityFunctionInterface):
        """Compute similarity between users (by age)"""
        def __init__(self, data_users, artworks_sim):
            self.data = data_users
            self.preprocess()

        def preprocess(self):
            self.age_index -= max(self.data['age']) - min (self.data['age'])

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFuntionInterface.computeSimilarity()"""
            ageA = self.data.loc[self.data['userId'] == A]['age'].to_list()[0]
            ageB = self.data.loc[self.data['userId'] == B]['age'].to_list()[0]
            return 1 - (abs(ageA - ageB) / (self.age_index))
            
    class SimilarityGender(SimilarityFunctionInterface):
        """Compute similarity between users (by age)"""
        def __init__(self, data_users, artworks_sim):
            self.data = data_users

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFuntionInterface.computeSimilarity()"""
            genderA = self.data.loc[self.data['userId'] == A]['gender'].to_list()[0]
            genderB = self.data.loc[self.data['userId'] == B]['gender'].to_list()[0]
            return genderA == genderB

    class SimilarityCountry(SimilarityFunctionInterface):
        """Compute similarity between users (by age)"""
        def __init__(self, data_users, artworks_sim):
            self.data = data_users

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFuntionInterface.computeSimilarity()"""
            countryA = self.data.loc[self.data['userId'] == A]['country'].to_list()[0]
            countryB = self.data.loc[self.data['userId'] == B]['country'].to_list()[0]
            return countryA == countryB

    class SimilarityRegion(SimilarityFunctionInterface):
        """Compute similarity between users (by age)"""
        def __init__(self, data_users, artworks_sim):
            self.data = data_users
            self.preprocess()

        def preprocess(self):
            """Replace country by its region"""
            regions = pd.read_csv(COUNTRIES_CSV).filter(items = ['Country', 'Region'])
            for country in self.data.country.unique():
                if country != 'Other':
                    self.data = self.data.replace(country, regions.loc[regions['Country'] == country]['Region'].to_list()[0])

        def computeSimilarity(self, A, B):
            """Overrides SimilarityFuntionInterface.computeSimilarity()"""
            regionA = self.data.loc[self.data['userId'] == A]['country'].to_list()[0]
            regionB = self.data.loc[self.data['userId'] == B]['country'].to_list()[0]
            return regionA == regionB
        
    class SimilarityPolarityPositive(SimilarityFunctionInterface):
        """Compute similarity between users (by artwork tastes)"""
        def __init__(self, data_users, artworks_sim):
            self.artworks_sim = artworks_sim
            self.data = data_users
            
        def computeSimilarity(self, A, B):
            """Overrides SimilarityFunctionInterface.computeSimilarity()"""
            listA = self.data.loc[self.data['userId'] == A]['positive'].to_list()[0]
            listB = self.data.loc[self.data['userId'] == B]['positive'].to_list()[0]
            sim = 0
            longest_list, shortest_list = (listA,listB) if len(listA) >= len(listB) else (listB,listA)
            for art1 in longest_list:
                max_sim = 0
                for art2 in shortest_list:
                    if self.artworks_sim[art1][art2] > max_sim:
                        max_sim = self.artworks_sim[art1][art2]
                sim += max_sim
            
            total_art = len(set(listA).union(set(listB))) 
            if total_art > 0:
                sim /= total_art
            elif total_art == 0: # Ambas listas están vacías
                sim = 1
            
            return sim
#             if set(listA) == set(listB):
#                 sim = 1
#             else:
#                 i = 0
#                 for art1 in listA:
#                     # max_pos = 0
#                     for art2 in listB:
#                         sim += self.artworks_sim[art1][art2]
#                         i += 1
#                 sim /= i if i > 0 else 1
            
            
    class SimilarityPolarityNegative(SimilarityFunctionInterface):
        """Compute similarity between users (by artwork tastes)"""
        def __init__(self, data_users, artworks_sim):
            self.artworks_sim = artworks_sim
            self.data = data_users
            
        def computeSimilarity(self, A, B):
            """Overrides SimilarityFunctionInterface.computeSimilarity()"""
            listA = self.data.loc[self.data['userId'] == A]['negative'].to_list()[0]
            listB = self.data.loc[self.data['userId'] == B]['negative'].to_list()[0]
            sim = 0
            longest_list, shortest_list = (listA,listB) if len(listA) >= len(listB) else (listB,listA)
            for art1 in longest_list:
                max_sim = 0
                for art2 in shortest_list:
                    if self.artworks_sim[art1][art2] > max_sim:
                        max_sim = self.artworks_sim[art1][art2]
                sim += max_sim
            
            total_art = len(set(listA).union(set(listB))) 
            if total_art > 0:
                sim /= total_art
            elif total_art == 0: # Ambas listas están vacías
                sim = 1
            
            return sim
        
    class SimilarityPolarityMixed(SimilarityFunctionInterface):
        """Compute similarity between users (by artwork tastes)"""
        def __init__(self, data_users, artworks_sim):
            self.artworks_sim = artworks_sim
            self.data = data_users
            
        def computeSimilarity(self, A, B):
            """Overrides SimilarityFunctionInterface.computeSimilarity()"""
            listA = self.data.loc[self.data['userId'] == A]['mixed'].to_list()[0]
            listB = self.data.loc[self.data['userId'] == B]['mixed'].to_list()[0]
            sim = 0
            longest_list, shortest_list = (listA,listB) if len(listA) >= len(listB) else (listB,listA)
            for art1 in longest_list:
                max_sim = 0
                for art2 in shortest_list:
                    if self.artworks_sim[art1][art2] > max_sim:
                        max_sim = self.artworks_sim[art1][art2]
                sim += max_sim
            
            total_art = len(set(listA).union(set(listB))) 
            if total_art > 0:
                sim /= total_art
            elif total_art == 0: # Ambas listas están vacías
                sim = 1
            
            return sim
