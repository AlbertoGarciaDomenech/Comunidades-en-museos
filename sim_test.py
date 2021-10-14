import pandas as pd
from setup import PATHS, PARAMS
from users_similarity import *
from artwork_similarity import *


sim = DepictsSimilarity(depth=PARAMS['DEPICTS_SIM_DEPTH'])  # depth=3|2|1
#sim = SizeSimilarity()
#sim = DominantColorSimilarity()
#sim = ArtistSimilarity()
#sim = ImageMSESimilarity()


# Cargamos los wd_paintingIDS de los cuadros
paintingIDS = pd.read_csv(PATHS['ARTWORKS_DATA'])['wd:paintingID'].unique()

# Recuperamos las similitudes parciales con los wd como argumento
print(sim.getSimilarity(paintingIDS[3], paintingIDS[1])) # recompute=True para recalcular la similitud obviando el contenido de la cache

sim.close()

