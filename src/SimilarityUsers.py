import pandas as pd
import numpy as np
import math
from src.SimilarityFunctionInterface import *

COUNTRIES_CSV = "data/countries_of_the_world.csv"

# USERS_CSV = 'data/Prado_users.csv'
# USERS_SCALED_CSV = 'data/Prado_users_scaled.csv'
# USERS_EMOTIONS_CSV = 'data/Prado_users_emotions_scaled_OnePolarity.csv'
# # USERS_EMOTIONS_SCALED_CSV = 'data/Prado_users_emotions_scaled.csv'

# ## USERS_INDIVIDUO_EXPLICADOR
# USERS_EMOTIONS_SCALED_CSV = 'data/Prado_users_individuo_explicador.csv'
# ##

class SimilarityAge(SimilarityFunctionInterface):
    """Compute similarity between users (by age)"""
    def __init__(self, data_users):
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
        return 1 - (1 / (self.age_index - 1) * abs(ageA - ageB))
    
class SimilarityGender(SimilarityFunctionInterface):
    """Compute similarity between users (by age)"""
    def __init__(self, data_users):
        self.data = data_users
      
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFuntionInterface.computeSimilarity()"""
        genderA = self.data.loc[self.data['userId'] == A]['gender'].to_list()[0]
        genderB = self.data.loc[self.data['userId'] == B]['gender'].to_list()[0]
        return genderA == genderB
    
class SimilarityCountry(SimilarityFunctionInterface):
    """Compute similarity between users (by age)"""
    def __init__(self, data_users):
        self.data = data_users
        
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFuntionInterface.computeSimilarity()"""
        countryA = self.data.loc[self.data['userId'] == A]['country'].to_list()[0]
        countryB = self.data.loc[self.data['userId'] == B]['country'].to_list()[0]
        return countryA == countryB

class SimilarityRegion(SimilarityFunctionInterface):
    """Compute similarity between users (by age)"""
    def __init__(self, data_users):
        self.data = data_users
        self.preprocess()
        
    def preprocess(self):
        """Replace country by its region"""
        regions = pd.read_csv(COUNTRIES_CSV).filter(items = ['Country', 'Region'])
        for country in self.data.country.unique():
            if country != 'Other':
                self.data = self.data.replace(country, regions.loc[regions['Country'] == country]['Region'])
        
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFuntionInterface.computeSimilarity()"""
        regionA = self.data.loc[self.data['userId'] == A]['country'].to_list()[0]
        regionB = self.data.loc[self.data['userId'] == B]['country'].to_list()[0]
        return regionA == regionB
    
###################################################
class SimilarityDemographic(SimilarityFunctionInterface):
    """Compute similarity between users (by demographic)"""
    def __init__(self, data_users, country_weight=0.3, age_weight=0.5, gender_weight=0.2):
        self.data_users = data_users
        self.country_weight = country_weight
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.countrySim = SimilarityRegion(self.data_users)
        self.ageSim = SimilarityAge(self.data_users)
        self.genderSim = SimilarityGender(self.data_users)
    
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFuntionInterface.computeSimilarity()"""
        country_sim = self.countrySim.computeSimilarity(A, B)
        age_sim = self.ageSim.computeSimilarity(A, B)
        gender_sim = self.genderSim.computeSimilarity(A, B)
                       
        return (country_sim*self.country_weight) + (age_sim*self.age_weight) + (gender_sim*self.gender_weight)
    
######################################################    
class SimilarityPolarity(SimilarityFunctionInterface):
    """Compute similarity between users (by artwork tastes)"""
    def __init__(self, data_users, artworks_sim, positive_weight=0.4, negative_weight=0.4, mixed_weight=0.2):
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.mixed_weight = mixed_weight
        self.artworks_sim = artworks_sim
        self.data = data_users
    
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        
        ###################### TODO: REVISAR ####################################
        # positiveA = self.data.loc[self.data['userId'] == A]['positive'].apply(eval).to_list()[0]
        # positiveB = self.data.loc[self.data['userId'] == B]['positive'].apply(eval).to_list()[0]
        # negativeA = self.data.loc[self.data['userId'] == A]['negative'].apply(eval).to_list()[0]
        # negativeB = self.data.loc[self.data['userId'] == B]['negative'].apply(eval).to_list()[0]
        # mixedA = self.data.loc[self.data['userId'] == A]['mixed'].apply(eval).to_list()[0]
        # mixedB = self.data.loc[self.data['userId'] == B]['mixed'].apply(eval).to_list()[0]
        positiveA = self.data.loc[self.data['userId'] == A]['positive'].to_list()[0]
        positiveB = self.data.loc[self.data['userId'] == B]['positive'].to_list()[0]
        negativeA = self.data.loc[self.data['userId'] == A]['negative'].to_list()[0]
        negativeB = self.data.loc[self.data['userId'] == B]['negative'].to_list()[0]
        mixedA = self.data.loc[self.data['userId'] == A]['mixed'].to_list()[0]
        mixedB = self.data.loc[self.data['userId'] == B]['mixed'].to_list()[0]
        
        
        positive_sim = 0
        if set(positiveA) == set(positiveB):
            positive_sim = 1
        else:
            i = 0
            for art1 in positiveA:
                # max_pos = 0
                for art2 in positiveB:
                    positive_sim += self.artworks_sim[art1][art2]
                    i += 1
            positive_sim /= i if i > 0 else 1
                    # max_pos = self.artworks_sim[art1][art2] if self.artworks_sim[art1][art2] > max_pos else max_pos
                # positive_sim += max_pos

            # max_aux = len(positiveA) if len(positiveA) > len(positiveB) else len(positiveB)
            # positive_sim = (positive_sim / max_aux) if max_aux > 0 else 1
        
        negative_sim = 0
        if set(negativeA) == set(negativeB):
            negative_sim = 1
        else:
            i = 0
            for art1 in negativeA:
                # max_neg = 0
                for art2 in negativeB:
                    negative_sim += self.artworks_sim[art1][art2]
                    i += 1
            negative_sim /= i if i > 0 else 1
                    # max_neg = self.artworks_sim[art1][art2] if self.artworks_sim[art1][art2] > max_neg else max_neg
                # negative_sim += max_neg

            # max_aux = len(negativeA) if len(negativeA) > len(negativeB) else len(negativeB)
            # negative_sim = (negative_sim / max_aux) if max_aux > 0 else 1
        
        mixed_sim = 0
        if set(mixedA) == set(mixedB):
            mixed_sim = 1
        else:
            i = 0
            for art1 in mixedA:
                # max_mix = 0
                for art2 in mixedB:
                    mixed_sim += self.artworks_sim[art1][art2]
                    i += 1
            mixed_sim /= i if i > 0 else 1
                    # max_mix = self.artworks_sim[art1][art2] if self.artworks_sim[art1][art2] > max_mix else max_mix
                # mixed_sim += max_mix
        
        # max_aux = len(mixedA) if len(mixedA) > len(mixedB) else len(mixedB)
        # mixed_sim = (mixed_sim / max_aux) if max_aux > 0 else 1
        
        return (positive_sim * self.positive_weight) + (negative_sim * self.negative_weight) + (mixed_sim * self.mixed_weight)
        
        
######################################################    
# class SimilarityUsers(SimilarityFunctionInterface):
#     """Compute similarity between users"""
#     def __init__(self, data_users, artworks_sim, demog_weight = 0.5, artw_weight = 0.5):
#         self.data_users = pd.read_csv(data_users)
#         self.demog_weight = demog_weight
#         self.artw_weight = artw_weight
#         self.artworks_sim = artworks_sim
        
#     def computeSimilarity(self, A, B):
#         """Overrides SimilarityFunctionInterface.computeSimilarity()"""
#         demog_sim = SimilarityDemographic(data_users).computeSimilarity(A, B)
#         artw_sim = SimilarityPolarity(self.artworks_sim).computeSimilarity(A, B)                                                
#         return (self.demog_weight * demog_sim) + (self.artw_weight * artw_sim)
    
    
########################---------------------------------------------------##############################
class SimilarityUsers(SimilarityFunctionInterface):
    """Compute similarity between users"""
    def __init__(self, data_users, artworks_sim, demog_weight = 0.5, artw_weight = 0.5):
        self.data_users = data_users
        self.demog_weight = demog_weight
        self.artw_weight = artw_weight
        self.artworks_sim = artworks_sim
        self.demogSim = SimilarityDemographic(self.data_users)
        self.polSim = SimilarityPolarity(self.data_users, self.artworks_sim)
    
    def getSimilarityMatrix(self):
        users_matrix = []
        for i in range(0, len(self.data_users)):
            sim_list = []
            for j in range(0, len(self.data_users)): 
                sim = self.computeSimilarity(self.data_users.loc(0)[i]['userId'], self.data_users.loc(0)[j]['userId'])
                if sim >= 0:
                    sim_list.append(sim)
                else:
                    sim_list.append(0)
            users_matrix.append(sim_list)
        
        return pd.DataFrame(users_matrix, index = [i for i in self.data_users['userId']], columns = [i for i in self.data_users['userId']])
        
    def computeSimilarity(self, A, B):
        """Overrides SimilarityFunctionInterface.computeSimilarity()"""
        demog_sim = self.demogSim.computeSimilarity(A, B)
        artw_sim = self.polSim.computeSimilarity(A, B)                                                
        return (self.demog_weight * demog_sim) + (self.artw_weight * artw_sim)