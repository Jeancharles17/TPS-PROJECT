import math
import random

class Individual:
    def __init__(self, chromosome, cities):
        self.chromosome = chromosome
        self.cities = cities
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        total_distance = 0
        num_cities = len(self.chromosome)
        for i in range(num_cities):
            current_city = self.chromosome[i]
            next_city = self.chromosome[(i + 1) % num_cities]  # Wrap around to the first city
            total_distance += self.calculate_distance(current_city, next_city)
        return 1 / total_distance  # Inverse of the total distance

    def calculate_distance(self, city1, city2):
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def generate_individual(num_cities):
    chromosome = list(range(num_cities))
    random.shuffle(chromosome)
    return chromosome


def generate_initial_population(num_individuals, num_cities, cities):
    population = []
    for _ in range(num_individuals):
        chromosome = generate_individual(num_cities)
        population.append(Individual(chromosome, cities))
    return population


def ordered_crossover(parent1, parent2):
    num_cities = len(parent1.chromosome)
    start = random.randint(0, num_cities - 1)
    end = random.randint(start + 1, num_cities)
    offspring_chromosome = [-1] * num_cities

    # Copy a portion from parent1 to offspring
    offspring_chromosome[start:end] = parent1.chromosome[start:end]

    # Fill the remaining genes with the order of parent2's genes
    parent2_index = 0
    for i in range(num_cities):
        if offspring_chromosome[i] == -1:
            while parent2.chromosome[parent2_index] in offspring_chromosome:
                parent2_index = (parent2_index + 1) % num_cities
            offspring_chromosome[i] = parent2.chromosome[parent2_index]
            parent2_index = (parent2_index + 1) % num_cities

    return Individual(offspring_chromosome, parent1.cities)


def swap_mutation(individual):
    chromosome = individual.chromosome.copy()
    index1 = random.randint(0, len(chromosome) - 1)
    index2 = random.randint(0, len(chromosome) - 1)
    chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
    return Individual(chromosome, individual.cities)


def select_parents(population):
    tournament_size = 3
    selected_parents = []
    for _ in range(2):
        participants = random.sample(population, tournament_size)
        winner = max(participants, key=lambda individual: individual.fitness)
        selected_parents.append(winner)
    return selected_parents


def replace_population(population, offspring):
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    num_offspring = len(offspring)
    population[-num_offspring:] = offspring
    return population


def terminate_condition(generations_without_improvement, max_generations):
    return generations_without_improvement >= max_generations

def genetic_algorithm(num_cities, cities, num_individuals, max_generations):
    population = generate_initial_population(num_individuals, num_cities, cities)
    best_fitness = 0
    generations_without_improvement = 0

    while not terminate_condition(generations_without_improvement, max_generations):
        parents = select_parents(population)
        offspring = [ordered_crossover(parents[0], parents[1]) for _ in range(num_individuals)]
        offspring = [swap_mutation(individual) for individual in offspring]
        population = replace_population(population, offspring)

        # Track the best fitness
        best_individual = max(population, key=lambda individual: individual.fitness)
        if best_individual.fitness > best_fitness:
            best_fitness = best_individual.fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

    best_individual = max(population, key=lambda individual: individual.fitness)
    best_route = [cities[city] for city in best_individual.chromosome]
    return best_route, best_individual.fitness


cities = [(0, 1), (2, 2), (3, 4), (6, 7)]
num_cities = len(cities)
num_individuals = 10
max_generations = 100

best_route, best_fitness = genetic_algorithm(num_cities, cities, num_individuals, max_generations)
print("Best route:", best_route)
print("Best fitness:", best_fitness)



