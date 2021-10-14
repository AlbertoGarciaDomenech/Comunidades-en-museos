import ast
import logging
import numpy as np
import pandas as pd
from setup import PATHS
from genetic_algorithm import GeneticAlgorithm
from artwork_similarity import ArtworkSimilarity, checkWeights

# Cargamos los conjuntos de datos relacionados con las respuestas de ususarios con respecto a la similitud
answers = pd.read_json(PATHS["SIMILARITY_ANSWERS"])[['userId', 'image1', 'image2', 'similarity']]
asked = pd.read_csv(PATHS["ASKED_USERS"])[['userCategory', 'userId']]
artworks = pd.read_csv(PATHS["ARTWORKS_DATA"])['wd:paintingID'].to_list()

# Extraemos las listas de profesionales y amateurs
asked['userCategory'] = asked['userCategory'].apply(lambda x : ast.literal_eval(x))
asked['userCategory'] = asked['userCategory'].apply(lambda x : x[0] if len(x) else None)
professionals = asked.loc[asked['userCategory'] == 'Professional']['userId'].to_list()
amateurs = asked.loc[asked['userCategory'] == 'Amateur']['userId'].to_list()

# Filtramos las respuestas de similitud por usuarios profesionales
answers = answers.loc[answers.userId.isin(amateurs)] #answers.userId.isin(professionals)|

# Filtramos por los cuadros presentes en nuestro dataset
answers = answers.loc[answers.image1.isin(artworks) & answers.image2.isin(artworks)]

# Descartamos la columna de userId
answers.drop(labels=['userId'], axis=1, inplace=True)

# Agrupamos por pares de cuadros iguales 
answers = answers.groupby(['image1', 'image2']).agg({'similarity' : ['max', 'min', 'mean']})
answers.columns = ['sim_max', 'sim_min', 'sim_mean']
answers = answers.reset_index()

logging.basicConfig(filename="data/log/weightSearchAmateurs.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s -> %(message)s')

using_sim = 'sim_min'

def fitness(x):
    error = .0
    for _, row in answers.iterrows():
        estimatedSim = ArtworkSimilarity(row['image1'], row['image2'], x) * 5
        error += abs(row[using_sim] - estimatedSim)
    return error / len(answers)

partials = 5
bounds = np.array([[1,10]]*partials)
params = {'iterations' : 100,
        'popSize' : 100,
        'crossoverProb' : 0.5,
        'mutationProb' : 0.2,
        'elitePerc' : 0.03,
        'crossoverMethod' : 'uniform'  }

gen = GeneticAlgorithm(fitness, partials, bounds, params)
gen.run()

results = gen.getResults()
best_ch = checkWeights(results['chromosome'], partials)
best_sc = results['value']

logging.info("Search ended using sim [" + using_sim + "] with WEIGHTS " + str(best_ch) + " and MEAN ERROR " + str(best_sc))