"""'
Пахомов Д.В.
Группа: 224-321
Почта: pakhomovdv0@gmail.com
Курс: https://online.mospolytech.ru/course/view.php?id=10055
Год разработки: 2023
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os


SEED = 15
random.seed(SEED)


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMin()


class FitnessMin:
    def __init__(self):
        self.values = [0]


class GA:
    def __init__(self,
                 adjacency_table,
                 start_point,
                 finish_point,
                 size_population,
                 mutation,
                 max_generation,
                 crossover=1):

        self.max_generation = max_generation
        self.mutation = mutation
        self.crossover = crossover
        self.size_population = size_population
        self.adjacency_table = adjacency_table
        self.start_point = start_point
        self.finish_point = finish_point
        self.len_graph = len(adjacency_table)
        self.len_chrom = self.len_graph * len(adjacency_table[0])

        self.minFitnessValues = []
        self.meanFitnessValues = []
        self.best_ind = []
        self.population = []
        self.generationCounter = 0
        self.list_population = []
        self.statistics_generation = []

        self.checkStep = 0
        self.log_step = ''

    def graph_fitness(self, individual):
        """Функция приспособленности"""
        _sum = 0
        way = individual[0].copy()
        n = self.finish_point
        way = way[:way.index(n) + 1]
        st = self.start_point
        if st in way:
            way.remove(st)
        for j in way:
            _sum += self.adjacency_table[st][j]
            st = j
        return _sum,

    def individual_create(self):
        """Генерация индивидуума"""
        return Individual([[i for i in range(self.len_graph)]])

    def population_create(self, n=0):
        """ Создание популяции"""
        return list([self.individual_create() for i in range(n)])

    @staticmethod
    def __clone(value):
        """Клонирование индивидуума"""
        ind = Individual(value[:])
        ind.fitness.values[0] = value.fitness.values[0]
        return ind

    @staticmethod
    def selTournament(population, p_len):
        offspring = []
        for n in range(p_len):
            i1 = i2 = i3 = 0
            while i1 == i2 or i1 == i3 or i2 == i3:
                i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

            offspring.append(
                min([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

        return offspring

    @staticmethod
    def mutantShuffle(mutant, indpb=1):
        for gen in mutant:
            if random.random() < indpb:
                start_p = random.randint(1, len(gen) - len(gen) // 2)
                end_p = start_p + random.randint(1, len(gen) - start_p)
                mutant_gen = gen[start_p: end_p]
                np.random.shuffle(mutant_gen)

                gen[start_p: end_p] = mutant_gen

    @staticmethod
    def cxOrdered(ind1_, ind2_):
        ind1, ind2 = ind1_[0].copy(), ind2_[0].copy()
        f = size = len(ind1)
        a = random.randint(1, size - f // 2)
        b = a + random.randint(1, size - a)

        temp1, temp2 = ind1, ind2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            ind1[k1] = temp1[(i + b + 1) % size]
            k1 += 1

            ind2[k2] = temp2[(i + b + 1) % size]
            k2 += 1

        for i in range(a, b):
            ind1[i], ind2[i] = ind2[i], ind1[i]

        return ind1, ind2

    @staticmethod
    def cxOrdered_v2(ind1_, ind2_):
        ind1, ind2 = ind1_[0].copy(), ind2_[0].copy()
        f = size = len(ind1)
        a = random.randint(1, size - f // 2)
        b = a + random.randint(1, size - a)
        temp1 = [-1 for _ in range(f)]
        temp2 = temp1[:].copy()

        for i in range(a, b):
            temp1[i], temp2[i] = ind2[i], ind1[i]
        j1 = j2 = b
        while -1 in temp1:

            for i in range(f):
                if ind1[i] in temp1:
                    continue
                if j1 >= f:
                    j1 = 0
                temp1[j1] = ind1[i]
                j1 += 1
        while -1 in temp2:
            for i in range(f):
                if ind2[i] in temp2:
                    continue
                if j2 >= f:
                    j2 = 0
                temp2[j2] = ind2[i]
                j2 += 1
        return [ind1], [ind2]

    def get_statistics(self):
        pass

    def createGA(self):
        self.population = self.population_create(n=self.size_population)
        self.generationCounter = 0
        fitnessValues = list(map(self.graph_fitness, self.population))

    def draw_statistics(self):
        path = "E:\\GitHub\\DoIS\\LW_1\\plot\\statistics.jpg"
        plt.plot(self.minFitnessValues, color='green')
        plt.plot(self.meanFitnessValues, color='blue')
        plt.xlabel('Поколение')
        plt.ylabel('Макс/средняя приспособленность')

        try:
            os.remove('LW_1/plot/state.png')
        except:
            pass
        path = 'E:\\GitHub\\DoIS\\LW_1\\plot\\state.png'
        plt.savefig(path)
        plt.close()

        return path

    def StepGA(self):
        if self.generationCounter < self.max_generation:
            if self.checkStep == 1:
                self.log_step = f'Поколение {self.generationCounter}'

            elif self.checkStep == 2:
                self.log_step = 'Селекция популяции'
                offspring = self.selTournament(self.population, len(self.population))
                self.population = list(map(self.__clone, offspring))
            elif self.checkStep == 3:
                self.log_step = 'Скрещивание популяции'

                new_offspring = []

                for child1, child2 in zip(self.population[::2], self.population[1::2]):
                    if random.random() < self.crossover:
                        child1, child2 = self.cxOrdered(child1, child2)
                    new_offspring.append(Individual(child1))
                    new_offspring.append(Individual(child2))
                self.population = new_offspring.copy()
            elif self.checkStep == 4:
                self.log_step = 'Мутация популяции'
                for mutant in self.population:
                    if random.random() < self.mutation:
                        self.mutantShuffle(mutant)
            if self.checkStep > 4:
                self.log_step = ''
                self.generationCounter += 1
                self.checkStep = 0

    def TrainGA(self):
        generationCounter = self.generationCounter
        print(f'---------------< {generationCounter} >--------------')
        population = self.population
        print(f'---------------< Начальная популяция >--------------')
        print(population)
        list(map(self.graph_fitness, self.population))
        while generationCounter < self.max_generation:
            self.list_population.append(population)
            generationCounter += 1
            print(f'---------------< {generationCounter} >--------------')

            offspring = self.selTournament(population, len(population))

            print(f'---------------< Популяция после селекции >--------------')
            print(offspring)

            offspring = list(map(self.__clone, offspring))
            new_offspring = []

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover:
                    child1, child2 = self.cxOrdered_v2(child1, child2)
                new_offspring.append(Individual(child1))
                new_offspring.append(Individual(child2))
            offspring = new_offspring.copy()

            print(f'---------------< Популяция после скрещивания >--------------')
            print(offspring)

            for mutant in offspring:
                if random.random() < self.mutation:
                    self.mutantShuffle(mutant)

            print(f'---------------< Популяция после мутации >--------------')
            print(offspring)

            freshFitnessValues = list(map(self.graph_fitness, offspring))
            for individual, fitnessValue in zip(offspring, freshFitnessValues):
                individual.fitness.values = fitnessValue

            population[:] = offspring
            fitnessValues = [ind.fitness.values[0] for ind in population]
            minFitness = min(fitnessValues)
            meanFitness = sum(fitnessValues) / len(population)
            self.minFitnessValues.append(minFitness)
            self.meanFitnessValues.append(meanFitness)

            best_index = fitnessValues.index(min(fitnessValues))
            self.best_ind = population[best_index]

            state = f"Поколение {generationCounter}: Минимальная приспособ = {minFitness}, Средняя приспособ= {meanFitness} \n Лучший индивидуум = {self.best_ind} "
            self.statistics_generation.append(state)

    def get_shortest_path(self):
        return self.best_ind[0]
