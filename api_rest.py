import csv
from decimal import Decimal

from sanic import Sanic
from sanic.response import json as sanjson
from setup import PATHS
from artwork_similarity import *
from sanic.response import file
from sanic.response import text
import os
import json
from sanic_cors import CORS, cross_origin




# Se instancia la aplicación Sanic
app = Sanic(name='api-rest')


# Permite CORS en todos los dominios
CORS(app)

# Restful API para listar todos los cuadros
@app.get('/artworks')
async def index(request):
    try:
        arts = []
        data = open(PATHS['ARTWORKS_DATA'])
        reader = csv.DictReader(data, delimiter=',')
        for row in reader:
            x = {
                "Title": row['Title'],
                "Id": row['wd:paintingID'],
                "Artist": row['Artist'],
                "Category": row['Category'],
                "Image": row['Image URL']
            }
            arts.append(x)
        return sanjson(arts)
    except:
        return sanjson({"Message": "Artworks not found"}, status=400)


# Restful API para listar los 5 cuadros más similares al que recibe, recibiendo (o no) pesos para las variables de similitud
@app.post('/artworks/similarity/artworkID')
async def index(request):
    try:
        id = request.args.get("id")
        weightsReq = []
        data = request.json
        if data:
            for it in data:
                weightsReq.append(float(Decimal(data.get(it))))

            x = Decimal(0)
            for i in weightsReq:
                x += Decimal(str(i))

            if x != 1.0:
                return sanjson({"Message": "Invalid weights"}, status=400)
            else:
                art = findSimilars(id, weightsReq)
                return sanjson(art)
        else:
            art = findSimilars(id, weightsReq)
            return sanjson(art)
    except:
        return sanjson({"Message": "Artwork ID not found"}, status=400)


# Restful API para mostrar archivo que contiene las comunidades visualmente
@app.get('/artworks/similarity/clusters')
async def index(request):
    try:
        return await file(PATHS['CLUSTER_VISUAL'])
    except:
        return sanjson({"Message": "There is no file for these clusters"}, status=400)


# Restful API para mostrar archivo que contiene las comunidades visualmente
@app.get('/artworks/similarity/clustersHTML')
async def index(request):
    try:
        out = []
        directory = PATHS['CLUSTERS_DIR']
        files = os.listdir(directory)
        for filename in files:
            with open(os.path.join(directory, filename)) as file:
                data = json.load(file)
                out.append(data)
        return sanjson(out)
    except:
        return sanjson({"Message": "Error displaying clusters"}, status=400)


# Método auxiliar para montar el JSON del cuadro recibido y sus similares
def findSimilars(id, weightsReq):
    artworks = kMostSimilarArtworks(id, k=5, weights=weightsReq)
    sims = []
    data = open(PATHS['ARTWORKS_DATA'])
    reader = csv.DictReader(data, delimiter=',')
    for i in artworks:
        for row in reader:
            if row['wd:paintingID'] == i[1]:
                x = {
                    "Title": row['Title'],
                    "Similarity": i[0],
                    "Category": row['Category'],
                    "Artist": row['Artist'],
                    "Image": row['Image URL']
                }
                sims.append(x)
                data.seek(0)
                break
    for row in reader:
        if row['wd:paintingID'] == id:
            art = {
                "Selected artwork": id,
                "Title": row['Title'],
                "Category": row['Category'],
                "Artist": row['Artist'],
                "Image": row['Image URL'],
                "Similar artworks":
                    sims
            }
    return art