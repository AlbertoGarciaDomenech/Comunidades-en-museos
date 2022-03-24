import abc
import pandas as pd

class SimilarityFunctionInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data_csv):
        """Initialize the class"""
        self.data = pd.read_csv(data_csv)
    
    @abc.abstractmethod
    def computeSimilarity(self, A, B) -> float:
        """Compute similarity between two lists"""
        raise NotImplementedError()