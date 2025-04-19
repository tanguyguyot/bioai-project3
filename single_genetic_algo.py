import random
import numpy as np
from table_creation import *
import pandas as pd

def generate_individual(length) -> str:
    """
    Generate a random individual with length genes.
    """
    individual: str = "".join(random.choice("01") for _ in range(length))
    return individual

def mutate(bitstring, mutation_rate=0.01):
    """
    Mutate an individual by flipping bits with a given mutation rate.
    """
    if random.random() < mutation_rate:
        index = random.randint(0, len(bitstring) - 1)
        bitstring = bitstring[:index] + str(1 - int(bitstring[index])) + bitstring[index + 1:]
    return bitstring

def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create a child.
    """
    index = random.randint(1, len(parent1) - 1)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]
    return child1, child2

def fitness(bitstring: str, lookup_table: pd.DataFrame) -> float:
    """
    Calculate the fitness of an individual based on the error table.
    The fitness is defined as the negative of the error.
    """
    fitness_value = lookup_table.loc[lookup_table['Binary representation'] == bitstring, 'Lookup value'].values[0]
    return fitness_value

def select_parents(population, lookup_table, tournament_size=6):
    """
    Select parents for the next generation using tournament selection.
    """
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: fitness(x, lookup_table), reverse=True)
    parent1 = tournament[0]
    parent2 = tournament[1]
    return parent1, parent2

def genetic_algorithm(lookup_table, features_amount, population_size=100, generations=100, tournament_size=6, mutation_rate=0.01):
    population = [generate_individual(features_amount) for _ in range(population_size)]
    
    # Generation loop
    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, lookup_table, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        population = new_population
        # Print the best fitness of the generation
        best_fitness = max(fitness(ind, lookup_table) for ind in population)
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")
    # Return the best individual from the final population
    best_individual = max(population, key=lambda x: fitness(x, lookup_table))
    best_fitness = fitness(best_individual, lookup_table)
    print(f"Best individual: {best_individual}, Fitness: {best_fitness}")
    return best_individual, best_fitness

if __name__ == "__main__":
    
    
    # Example usage : glass dataset, it has 9 features
    lookup_table = pd.read_csv("glass_complete_table.csv", dtype={'Binary representation': str})
    lookup_table['Binary representation'] = lookup_table['Binary representation'].astype("string")

    best_individual, best_fitness = genetic_algorithm(lookup_table, features_amount=9)
    print(f"Best individual: {best_individual}, Fitness: {best_fitness}")