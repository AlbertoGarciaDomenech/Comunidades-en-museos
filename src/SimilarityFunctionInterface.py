from src.Singleton import *
import pandas as pd

class SimilarityFunctionInterface(metaclass=Singleton):
    
    def computeSimilarity(self, A, B) -> float:
        """Compute similarity between two lists"""
        raise NotImplementedError()