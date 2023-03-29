import numpy as np
import math
import os
import random

# Константы для ГА
SIZE_POPULATION = 200  # кол-во инд в популяции
CROSSOVER = 0.9  # вероятность селекции
MUTATION = 0.1  # вероятность мутации
MAX_GEN = 50  # максимально кол-во поколений
# Граф

inf = 100
Graph = ((0, 3, 1, 3, inf, inf),
         (3, 0, 4, inf, inf, inf),
         (1, 4, 0, inf, 7, 5),
         (3, inf, inf, 0, inf, 2),
         (inf, inf, 7, inf, 0, 4),
         (inf, inf, 5, 2, 4, 0))
START_POINT = 0
LENGraph = len(Graph)
LEN_CHROM = LENGraph * len(Graph[0])
SEED = 15
random.seed(SEED)


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMin()


class FitnessMin:
    def __init__(self):
        self.values = [0]


def graph_fitness(individual):
    _sum = 0
    for n, way in enumerate(individual):
        way = way[:way.index(n) + 1]

        st = START_POINT
        for j in way:
            _sum += Graph[st][j]
            st = j
