import abc

class SimilarityFunctionInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def computeSimilarity(self, A, B) -> float:
        """Compute similarity between two lists"""
        raise NotImplementedError()