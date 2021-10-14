from fcache.cache import FileCache
from setup import PATHS

class CachedSimilarity(object):

    def __init__(self, appname, fullname, shortName, cache_dir=PATHS['CACHE']):
        self._cacheFile_ = FileCache('communities.' + appname, flag='cs', app_cache_dir=cache_dir)
        self._fullName_ = fullname
        self._shortName_ = shortName

    def getFullName(self):
        return self._fullName_
    
    def getShortName(self):
        return self._shortName_

    def computeSimilarity(self, A, B):
        raise NotImplementedError("Each subclass must implement this method")

    def __findSimilarity(self, A, B):
        if A in self._cacheFile_ and B in self._cacheFile_[A].keys():
            return self._cacheFile_[A][B], (A, B)
        if B in self._cacheFile_ and A in self._cacheFile_[B].keys():
            return self._cacheFile_[B][A], (B, A)
        return None, None    

    def __lenDict(self, Entity):
        if Entity not in self._cacheFile_:
            self._cacheFile_[Entity] = dict()
        return len(self._cacheFile_[Entity])
    
    def __storeSim(self, A, B, sim):
        len_a = self.__lenDict(A)
        len_b = self.__lenDict(B)
        if len_a < len_b:
            self._cacheFile_[A] |= {B : sim}
        else:
            self._cacheFile_[B] |= {A : sim}

    def getSimilarity(self, A, B, recompute=False):
        sim, coord = self.__findSimilarity(A, B)
        if sim is None or recompute:
            sim = self.computeSimilarity(A, B)
            if coord is None:
                self.__storeSim(A, B, sim)
            else:
                self._cacheFile_[coord[0]] |= { coord[1] : sim }
        return sim

    def close(self):
        self._cacheFile_.sync()
        self._cacheFile_.close()
