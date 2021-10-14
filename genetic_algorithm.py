import numpy as np
import random as rd
from math import inf 
from joblib import Parallel, delayed

class GeneticAlgorithm(object):
    
    def __init__(self, fitness, chromosomeSize, boundaries, algorithm_parameters):
        
        assert chromosomeSize == len(boundaries), "Boundaries dimension must be the same than chromosome size"

        # Set constructor params
        self.fitness = fitness
        self.chromosomeSize = chromosomeSize
        self.boundaries = boundaries
        self.params = {'iterations' : 100,
                       'popSize' : 100,
                       'crossoverProb' : 0.3,
                       'mutationProb' : 0.1,
                       'elitePerc' : 0.01,
                       'crossoverMethod' : 'uniform'  } # or onePoint/twoPoint
        self.params |= algorithm_parameters

        self.numElite = max(round(self.params['popSize'] * self.params['elitePerc']), 1) # At least 1

        self.bestScore = inf
        self.bestChromosome = None

    
    def run(self):

        # Generate initial random population
        self.population = np.random.uniform(low=self.boundaries[:,0], 
                                            high=self.boundaries[:,1], 
                                            size=(self.params['popSize'], len(self.boundaries)))

        # Run genetic for params['iterations']
        for it in range(self.params['iterations']):

            # EVALUATE population with fitness function
            self.scores = np.array(Parallel(n_jobs=-1)(delayed(self.fitness)(c) for c in self.population))
            scoresSorted = list(zip(self.scores, np.arange(self.params['popSize'])))
            scoresSorted.sort()
            self.scores = np.array(list(zip(*scoresSorted))[0])
            self.population = self.population[np.array(list(zip(*scoresSorted))[1])]

            if self.scores[0] < self.bestScore:
                self.bestScore = self.scores[0]
                self.bestChromosome = self.population[0].copy()

            print("Generation ", it," -> [Best chromosome ", str(self.bestChromosome), " got score " + str(self.bestScore), "]")
            
            # Isolate ELITE
            elite = self.population[:self.numElite].copy()

            # SELECTION OF NEXT GENERATION (roullette by fitness)
            choice = np.random.choice(a=self.params['popSize'], size=self.params['popSize'], p = np.flip(self.scores / sum(self.scores)))
            self.population = self.population[choice]
            self.scores = self.scores[choice]            

            # CROSSOVER
            randomSelection = np.random.rand(self.params['popSize']) # Generamos un array de tamaño popSize de valores entre 0 y 1
            parents = self.population[np.where(randomSelection <= self.params['crossoverProb'])[0]].copy() # Seleccionamos como padres los que sean menores de crossoverProb
            nextGen = self.population[np.where(randomSelection > self.params['crossoverProb'])].copy() # Insertamos en la nueva generación los demás individuos no seleccionados
            nextGen = np.concatenate((elite, nextGen[:(-1 * self.numElite)])) # Reinsertamos la elite en la nueva generación eliminando los peores individuos para mantener el mismo tamaño de la población

            if (len(parents) % 2 == 1): # Si el número de padres es impar, sacamos uno y lo insertamos en la proxima generación 
                nextGen = np.concatenate((nextGen, parents[-1].reshape(1,-1)))
                parents = parents[:-1]
            
            for p in range(0, len(parents), 2):
                offspring = self.cross(parents[p],
                                       parents[p+1])
                nextGen = np.concatenate((nextGen, offspring))

            assert len(nextGen) == self.params['popSize']

            self.population = nextGen

            # MUTATION
            for i in range(self.numElite, self.population.shape[0]):
                for j in range(self.population.shape[1]):
                    if rd.uniform(0., 1.) <= self.params['mutationProb']:
                        self.population[i][j] = rd.uniform(self.boundaries[j][0], self.boundaries[j][1])

        if self.scores[0] < self.bestScore:
            self.bestScore = self.scores[0]
            self.bestChromosome = self.population[0].copy()

    def getResults(self):
        return {'chromosome'    :   self.bestChromosome,
                'value'         :   self.bestScore}

    def cross(self, X, Y):
         
        offs1 = X.copy()
        offs2 = Y.copy()
        

        if self.params['crossoverMethod'] == 'onePoint':
            ran = np.random.randint(0,self.chromosomeSize)
            for i in range(0,ran):
                offs1[i] = Y[i].copy()
                offs2[i] = X[i].copy()
  
        if self.params['crossoverMethod'] == 'twoPoint':
                
            ran1 = np.random.randint(0,self.chromosomeSize)
            ran2 = np.random.randint(ran1,self.chromosomeSize)
                
            for i in range(ran1,ran2):
                offs1[i] = Y[i].copy()
                offs2[i] = X[i].copy()
            
        if self.params['crossoverMethod'] == 'uniform':
                
            for i in range(0, self.chromosomeSize):
                ran = np.random.random()
                if ran < 0.5:
                    offs1[i] = Y[i].copy()
                    offs2[i] = X[i].copy() 
                   
        return np.array([offs1,offs2])

