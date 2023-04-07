import numpy as np

import random
import matplotlib.pyplot as plt

# Константы для ГА
SIZE_POPULATION = 300  # кол-во инд в популяции
CROSSOVER = 0.9  # вероятность селекции
MUTATION = 0.01  # вероятность мутации
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

minFitnessValues = []
meanFitnessValues = []


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
    return _sum,


def individual_create():
    """Генерация индивидуума"""
    return Individual(random.sample(range(LENGraph), LENGraph) for i in range(LENGraph))


def population_create(n=0):
    """ Создание популяции"""
    return list([individual_create() for i in range(n)])


population = population_create(n=SIZE_POPULATION)
generationCounter = 0
fitnessValues = list(map(graph_fitness, population))


def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        offspring.append(min([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring


def mutantShuffle(mutant, indpb=0.01):
    for gen in mutant:
        if random.random() < indpb:
            start_p = random.randint(1, len(gen) - 3)
            end_p = start_p + random.randint(1, len(gen) - start_p)
            mutant_gen = gen[start_p: end_p]
            np.random.shuffle(mutant_gen)

            gen[start_p: end_p] = mutant_gen


def cxOrdered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    a = random.randint(1, size - 3)
    b = a + random.randint(1, size - a)

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        ind1[k1 % size] = temp1[(i + b + 1) % size]
        k1 += 1

        ind2[k2 % size] = temp2[(i + b + 1) % size]
        k2 += 1

    for i in range(a, b):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


while generationCounter < SIZE_POPULATION:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))
    new_offspring = []
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER:
            child1, child2 = cxOrdered(child1, child2)
        new_offspring.append(Individual(child1))
        new_offspring.append(Individual(child2))
    offspring = new_offspring.copy()

    for mutant in offspring:
        if random.random() < MUTATION:
            mutantShuffle(mutant, indpb=1. / LEN_CHROM / 36)

    freshFitnessValues = list(map(graph_fitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring
    fitnessValues = [ind.fitness.values[0] for ind in population]
    minFitness = min(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    minFitnessValues.append(minFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {minFitness}, Средняя приспособ.= {meanFitness}")
    best_index = fitnessValues.index(min(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")

from matplotlib.lines import Line2D

vertex = ((0, 1), (1, 1), (0.5, 0.8), (0.1, 0.5), (0.8, 0.2), (0.4, 0))

vx = [v[0] for v in vertex]
vy = [v[1] for v in vertex]

best = population[best_index]
print(best)


def show_graph(ax, best):
    ax.add_line(Line2D((vertex[0][0], vertex[1][0]), (vertex[0][1], vertex[1][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[0][0], vertex[2][0]), (vertex[0][1], vertex[2][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[0][0], vertex[3][0]), (vertex[0][1], vertex[3][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[1][0], vertex[2][0]), (vertex[1][1], vertex[2][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[2][0], vertex[5][0]), (vertex[2][1], vertex[5][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[2][0], vertex[4][0]), (vertex[2][1], vertex[4][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[3][0], vertex[5][0]), (vertex[3][1], vertex[5][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[4][0], vertex[5][0]), (vertex[4][1], vertex[5][1]), color='#aaa'))

    startV = 0
    for i, v in enumerate(best):
        if i == 0:
            continue

        prev = startV
        v = v[:v.index(i) + 1]
        for j in v:
            ax.add_line(Line2D((vertex[prev][0], vertex[j][0]), (vertex[prev][1], vertex[j][1]), color='r'))
            prev = j

    ax.plot(vx, vy, ' ob', markersize=15)


plt.plot(minFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')

fig, ax = plt.subplots()
show_graph(ax, best)
plt.show()
