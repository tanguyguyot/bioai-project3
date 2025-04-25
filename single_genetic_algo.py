import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from table_creation import csv_to_dict

def generate_individual(length: int) -> str:
    """
    Generate a random individual with length genes.
    """
    individual: str = "".join(random.choice("01") for _ in range(length))
    return individual

def mutate(bitstring: str, mutation_rate: float) -> str:
    """
    Mutate an individual by flipping bits with a given mutation rate.
    """
    out = []
    for bit in bitstring:
        if random.random() < mutation_rate:
            out.append("1" if bit == "0" else "0")
        else:
            out.append(bit)
    return "".join(out)

def crossover(parent1: str, parent2: str) -> tuple:
    """
    Perform crossover between two parents to create a child.
    """
    index = random.randint(1, len(parent1) - 1)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]
    return child1, child2

def fitness(bitstring: str, lookup_dict: dict) -> float:
    """
    Calculate the fitness of an individual based on the error table.
    The fitness is defined as the negative of the error.
    """
    fitness_value = lookup_dict.get(bitstring, np.inf).get("Lookup value", np.inf)
    return fitness_value

def select_parents(population: list, lookup_dict: dict, tournament_size: int) -> tuple:
    """
    Select parents for the next generation using tournament selection.
    """
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: fitness(x, lookup_dict))
    parent1 = tournament[0]
    parent2 = tournament[1]
    return parent1, parent2

def elite_selection(population: list, lookup_dict: dict, amount: int) -> list:
    """
    Select the best individuals from the population based on fitness. Keep some individuals for diversity
    """
    population.sort(key=lambda x: fitness(x, lookup_dict))
    # Select the best individuals
    elite_population = population[:amount]
    # Add other individuals from the population at random to keep diversity
    remaining_population = population[amount:]
    random.shuffle(remaining_population)
    # Select the remaining individuals randomly
    remaining_amount = len(population) - amount
    return elite_population + remaining_population[:remaining_amount]

def genetic_algorithm(lookup_dict, population_size=1000, generations=1000, tournament_size=6, mutation_rate=0.01, elite_frac=0.2) -> tuple:
    features_amount = len(next(iter(lookup_dict.keys())))
    population = [generate_individual(features_amount) for _ in range(population_size)]

    begin = time.time()
    best_fitness = min(fitness(ind, lookup_dict) for ind in population)
    print(f"Initial best fitness: {best_fitness}")
    # Generation loop
    for generation in tqdm(range(generations), desc="Gen"):
        if generation % 100 == 0:
            tqdm.write(f"Generation {generation} ; best = {best_fitness}")
        children = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, lookup_dict, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            children += [mutate(child1, mutation_rate), mutate(child2, mutation_rate)]
        # Selection of the best individuals
        population = elite_selection(children, lookup_dict, amount=round(population_size * elite_frac))
        # Print the best fitness of the generation
        best_fitness = min(fitness(ind, lookup_dict) for ind in population)
    # Return the best individual from the final population
    best_individual = min(population, key=lambda x: fitness(x, lookup_dict))
    best_fitness = fitness(best_individual, lookup_dict)
    print(f"Best individual: {best_individual}, Fitness: {best_fitness}")
    # Save the last population to a CSV file, with all the scores and the amount of individuals
    population_df = pd.DataFrame(population)
    population_df['Fitness'] = [fitness(ind, lookup_dict) for ind in population]
    #population_df.to_csv("outputs/final_population_sga.csv", index=False)
    #print("Final population saved to outputs/final_population_sga.csv")
    end = time.time()
    print(f"Time taken: {end - begin} seconds for {generations} generations of {population_size} individuals")
    return best_individual, best_fitness

if __name__ == "__main__":
    # Dataset features amount : glass = 9, wine = 13, magic = 10
    glass_lookup_dict = csv_to_dict("outputs/glass_complete_table.csv")
    wine_lookup_dict = csv_to_dict("outputs/wine_complete_table.csv")
    magic_lookup_dict = csv_to_dict("outputs/magic_complete_table.csv")
    heart_lookup_dict = csv_to_dict("outputs/heart_diseases_complete_table.csv")
    zoo_lookup_dict = csv_to_dict("outputs/zoo_complete_table.csv")

    glass_best_individual, glass_best_fitness = genetic_algorithm(glass_lookup_dict)
    wine_best_individual, wine_best_fitness = genetic_algorithm(wine_lookup_dict)
    magic_best_individual, magic_best_fitness = genetic_algorithm(magic_lookup_dict)
    heart_best_individual, heart_best_fitness = genetic_algorithm(heart_lookup_dict)
    zoo_best_individual, zoo_best_fitness = genetic_algorithm(zoo_lookup_dict)
    print(f"Best individual for glass: {glass_best_individual}, Fitness: {glass_best_fitness}")
    print(f"Best individual for wine: {wine_best_individual}, Fitness: {wine_best_fitness}")
    print(f"Best individual for magic: {magic_best_individual}, Fitness: {magic_best_fitness}")
    print(f"Best individual for heart: {heart_best_individual}, Fitness: {heart_best_fitness}")
    print(f"Best individual for zoo: {zoo_best_individual}, Fitness: {zoo_best_fitness}")
    
    