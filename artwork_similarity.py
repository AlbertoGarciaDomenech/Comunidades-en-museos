
import math
import heapq
import colorsys
import numpy as np
import pandas as pd
import imageio as imio

from skimage import io
from collections import Counter
from setup import PATHS, PARAMS
from skimage import img_as_float
from sklearn.cluster import KMeans
from collections import OrderedDict
from skimage.transform import resize
from joblib import Parallel, delayed
from my_sparql import PropertyRetreiver
from cached_similarity import CachedSimilarity
from skimage.metrics import normalized_root_mse


_cos_sim = lambda x, y : round(x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))), 5)

def _minPath(a, b):
    if sum(a) < sum(b) or (sum(a) == sum(b) and max(a) <= max(b)):
        return a
    return b

def _findLeastCommonSubsumer(entity_a, entity_b, ret, max_depth=1):  
    '''
    Busca el antepasado común más cercano de dos entidades en la profundidad máxima indicada

    PARÁMETROS
        entity_a, entity_b  entidades para las que se busca antepasado común
        ret                 recuperador de superclases del tipo my_sparql.PropertyRetriever
        max_depth           profundidad máxima de la búsqueda
    DEVUELVE
        entities    tupla con las entidades objeto de la búsqueda
        lcs         diccionario con antepasados comunes más cercanos y las distancias de cada entidad a ellos. None si no se encuentra ninguno
    '''
    # Estructuras auxiliares para almacenar los antepasados encontrados y en que profundidad se hizo 
    depths_a = { entity_a : 0 }
    depths_b = { entity_b : 0 }

    for i in range(0, max_depth + 1):   
        # Si existe una intersección entre los conjuntos de antepasados encontrados
        intersection = set([*depths_a]) & set([*depths_b])  
        if len(intersection):
            # Devolvemos el antepasado con camino más corto de todos los comunes encontrados
            min_path = (math.inf, math.inf)
            for common in intersection:
                min_path = _minPath(min_path, (depths_a.get(common), depths_b.get(common)))
            lcs = dict()
            for common in intersection:
                cur_path = (depths_a.get(common), depths_b.get(common))
                if cur_path == min_path:
                    lcs |= { common : min_path}
                if cur_path == min_path[::-1]:  
                   lcs |= { common : min_path[::-1]}
            return (entity_a, entity_b), lcs
        # Si no añadimos a las estructuras los antepasados de los elementos recuperados en la profundidad anterior
        else:
            for sa in [x for x in [*depths_a] if depths_a.get(x) == i]:
                depths_a |= dict([(k, min(i+1, depths_a.get(k, math.inf))) for k in ret.retrieveFor(sa)])
            for sb in [y for y in [*depths_b] if depths_b.get(y) == i]:
                depths_b |= dict([(k, min(i+1, depths_b.get(k, math.inf))) for k in ret.retrieveFor(sb)])
    return (entity_a, entity_b), None

class DepictsSimilarity(CachedSimilarity):

    def __init__(self, depth=1, cache_dir=PATHS['CACHE']):
        super().__init__('artworkSimilarity.depicts.depth' + str(depth), 'depicts', 'dep',  cache_dir)
        self.__superclassRetreiver__ = PropertyRetreiver(['P279', 'P31'])
        self.__depictsRetreiver__ = PropertyRetreiver(['P180'])
        self.__maxdepth__ = depth
    
    # overriding abstract method
    def computeSimilarity(self, A, B):
        a = set(self.__depictsRetreiver__.retrieveFor(A))
        b = set(self.__depictsRetreiver__.retrieveFor(B))
        intersection = a & b
        exclusive_A = a - b
        exclusive_B = b - a

        discards = set()
        commons = dict()
        # Buscamos superclases comunes para todos los depicts de A y B
        for depicts, lcs in Parallel(n_jobs=-1)(delayed(_findLeastCommonSubsumer)(da, db, self.__superclassRetreiver__, self.__maxdepth__) for da in exclusive_A for db in exclusive_B):  
            if lcs is not None:
                discards |= {depicts[0], depicts[1]}
                commons |= {c : _minPath(commons.get(c, (math.inf, math.inf)), p) for c, p in lcs.items()}

        exclusive_A -= discards
        exclusive_B -= discards
        weights_a = OrderedDict({k : 1. for k in intersection} | {k : 1 / p[0] if p[0] else 1 for k, p in commons.items()} | {k : 1. for k in exclusive_A} | {k : 0. for k in exclusive_B})
        weights_b = OrderedDict({k : 1. for k in intersection} | {k : 1 / p[1] if p[1] else 1 for k, p in commons.items()} | {k : 0. for k in exclusive_A} | {k : 1. for k in exclusive_B})

        # Tomamos ambos vectores de pesos y calculamos la distancia del coseno entre ellos
        return _cos_sim(np.array([*weights_a.values()]), np.array([*weights_b.values()]))
    
    # overriding method to close both retreivers to
    def close(self):
        super().close()
        self.__superclassRetreiver__.close()
        self.__depictsRetreiver__.close()


class SizeSimilarity(CachedSimilarity):

    def __init__(self, cache_dir=PATHS['CACHE']):
        super().__init__('artworkSimilarity.picture.size', 'size', 'sz', cache_dir)
        self.__heightRetriever__ = PropertyRetreiver(['P2048'])
        self.__widthRetriever__ = PropertyRetreiver(['P2049'])

    def __computeArea(self, Entity):
        width = self.__widthRetriever__.retrieveFor(Entity)
        height = self.__heightRetriever__.retrieveFor(Entity)
        if width == [] or height == []:
            return 0.
        return float(width[0]) * float(height[0])

    def computeSimilarity(self, A, B):
        area1 = self.__computeArea(A)
        area2 = self.__computeArea(B)
        return min(area1, area2) / max(area1, area2)

    def close(self):
        super().close()
        self.__heightRetriever__.close()
        self.__widthRetriever__.close()


class DominantColorSimilarity(CachedSimilarity):

    def __init__(self, artworks_CSV=PATHS['ARTWORKS_DATA'], cache_dir=PATHS['CACHE']):
        super().__init__('artworkSimilarity.picture.dominantColor', 'dominantColor', 'dom', cache_dir)
        self.__pics__ = pd.read_csv(artworks_CSV)[['wd:paintingID', 'Image URL']] 

    def __colorPercentage(self, cluster):
        n_pixels = len(cluster.labels_)
        counter = Counter(cluster.labels_)  # count how many pixels per cluster
        perc = {}
        # Crea el vector de % dividiendo el numero de pixels de un cluster con el total
        for i in counter:
            perc[i] = np.round(counter[i] / n_pixels, 2)
        perc = dict(sorted(perc.items()))
        return perc

    def __dominantColor(self, entity):
        # Lee imagen
        url = self.__pics__.loc[self.__pics__['wd:paintingID'] == entity]['Image URL'].to_list()[0]
        img = imio.imread(url)
        # Hace Kmeans
        clt = KMeans(n_clusters=3, random_state=PARAMS['RANDOM_STATE'])
        clt_1 = clt.fit(img.reshape(-1, 3))
        # Crea el array de porcentajes de color
        perc = self.__colorPercentage(clt_1)
        # Indice del color dominante
        max_color_key = max(perc, key=perc.get)
        # Valores RGB del color dominante
        max_color_rgb = clt_1.cluster_centers_[max_color_key]
        # RGB to HSV
        return colorsys.rgb_to_hsv(max_color_rgb[0], max_color_rgb[1], max_color_rgb[2])

    def computeSimilarity(self, A, B):
        a = self.__dominantColor(A)
        b = self.__dominantColor(B)
        dh = min(abs(a[0]-b[0]), 360-abs(a[0]-b[0])) / 180.0
        ds = abs(a[1] - b[1])
        dv = abs(a[2] - b[2]) / 255.
        distance = math.sqrt(dh * dh + ds * ds + dv * dv)
        return round(1. - (distance), 2)


class ArtistSimilarity(CachedSimilarity):

    def __init__(self, artworks_CSV=PATHS['ARTWORKS_DATA'], cache_dir=PATHS['CACHE']):
        super().__init__('artworkSimilarity.artist', 'artist', 'art', cache_dir)
        self.__artist__ = pd.read_csv(artworks_CSV)[['wd:paintingID', 'Artist', 'Category']] 

    def __getArtist(self, entity):
        return self.__artist__.loc[self.__artist__['wd:paintingID'] == entity]['Artist'].to_list()[0]
        
    def __getCategory(self, entity):    
        return self.__artist__.loc[self.__artist__['wd:paintingID'] == entity]['Category'].to_list()[0]

    def computeSimilarity(self, A, B):
        if self.__getArtist(A) == self.__getArtist(B):
            return 1.
        if self.__getCategory(A) == self.__getCategory(B):
            return .85
        return .0   

    def close(self):
        super().close()


class ImageMSESimilarity(CachedSimilarity):

    def __init__(self, artworks_CSV=PATHS['ARTWORKS_DATA'], cache_dir=PATHS['CACHE']):
        super().__init__('artworkSimilarity.picture.mse_sim', 'imageMSE', 'mse', cache_dir)
        self.__pics__ = pd.read_csv(artworks_CSV)[['wd:paintingID', 'Image URL']] 

    def __mse_ssim(self, url1, url2):

        img1 = img_as_float((io.imread(url1)))
        img2 = img_as_float((io.imread(url2)))

        if (img1.shape != img2.shape):
            max0 = max(img1.shape[0], img2.shape[0])
            max1 = max(img1.shape[1], img2.shape[1])
            img2 = resize(img2, (max0, max1))
            img1 = resize(img1, (max0, max1))

        mse = (normalized_root_mse(img1, img2, normalization='min-max') + normalized_root_mse(img2, img1, normalization='min-max')) / 2

        return round((1 - mse), 5)

    
    def computeSimilarity(self, A, B):
        url1 = self.__pics__.loc[self.__pics__['wd:paintingID'] == A]['Image URL'].to_list()[0]
        url2 = self.__pics__.loc[self.__pics__['wd:paintingID'] == B]['Image URL'].to_list()[0]
        return self.__mse_ssim(url1, url2)


Partial_Similarities = [DepictsSimilarity(PARAMS['DEPICTS_SIM_DEPTH']),
                        SizeSimilarity(),
                        DominantColorSimilarity(),
                        ArtistSimilarity(),
                        ImageMSESimilarity()]

PradoArtworks = pd.read_csv(PATHS['ARTWORKS_DATA'])

def checkWeights(weights, length):
    if not len(weights) or sum(weights) == 0: # Si el vector está vacío devolvemos un vector con todos los pesos iguales
        return np.ones(length) * (1 / length)
    else: # Si no está vacío
        weights = np.array(weights)
        if len(weights) < length:   # Si es más corto que la longitud especificada, lo rellenamos con ceros
            weights = np.concatenate((np.array(weights), np.zeros(length - len(weights))))
        if length < len(weights):   # Si es más largo que la longitud especificada, quitamos los valores que exceden por el final
            weights = weights[:length] + (sum(weights[length-len(weights):]) / length)
        if (weights < 0).sum(): # Si hay algún valor negativo, desplazamos todos los valores sumándole el valor absoluto máximo del vector, quedando así todos en positivo
            weights += abs(weights).max()
        return weights / weights.sum()  # Nos aseguramos de que los pesos sumen 1. dividiendolos por su suma acumulada

def ArtworkSimilarity(A, B, weights=[]):
    weights = checkWeights(weights, len(Partial_Similarities))
    partials = []
    for partial in Partial_Similarities:
        partials.append(partial.getSimilarity(A, B))
    return (np.array(partials) * weights).sum()

def kMostSimilarArtworks(artwork, k=5, weights=[]):
    q = []
    for row in PradoArtworks['wd:paintingID'].unique():
        heapq.heappush(q, (ArtworkSimilarity(artwork, row, weights), row))
    return heapq.nlargest(k+1, q)[1:]

def mostSimilarArtworks(artwork, threshold=.75, weights=[]):
    q = []
    for row in PradoArtworks['wd:paintingID'].unique():
        sim = ArtworkSimilarity(artwork, row, weights)
        if sim >= threshold:
            heapq.heappush(q, (sim, row))
    return heapq.nlargest(len(q), q)[1:]
