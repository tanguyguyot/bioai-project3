import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from table_creation import csv_to_dict

# NSGA-II
def generate_individual(length: int) -> str:
    """
    Generate a random individual with length genes.
    """
    individual: str = "".join(random.choice("01") for _ in range(length))
    return individual

# Functions to minimize, objectives

def function1(bitstring: str, lookup_dict: dict) -> float:
    """
    Get the error value of the individual from the lookup table.
    """
    fitness_value = lookup_dict.get(bitstring, np.inf).get("Error", np.inf)
    return fitness_value

def function2(bitstring: str, lookup_dict: dict) -> float:
    """
    Get the number of features of the bitstring
    """
    return bitstring.count("1")

def non_dominance_condition(ind1_idx: str, ind2_idx: str, objectives: list) -> bool:
    """
    Check if individual1 dominates individual2.
    """
    f1_ind1, f2_ind1 = objectives[ind1_idx]
    f1_ind2, f2_ind2 = objectives[ind2_idx]
    
    # Condition 1 : function1 and function2 of ind1 are both less than or equal to ind2
    condition1 = all(x <= y for x, y in zip((f1_ind1, f2_ind1), (f1_ind2, f2_ind2)))
    # Condition 2 : at least one of them is strictly less than ind2
    condition2 = any(x < y for x, y in zip((f1_ind1, f2_ind1), (f1_ind2, f2_ind2)))
    return condition1 and condition2

def fast_non_dominated_sort(objectives: list) -> list:
    """
    Perform fast non-dominated sorting on the population.
    """
    len_population = len(objectives)
    fronts = [[]]
    # List of solutions dominated by each individual (all q that are dominated by p)
    dominated = [[] for _ in range(len_population)]
    # number of solutions dominating each individual (nbr of q that dominate dominating[p])
    dominating = [0 for _ in range(len_population)]
    
    # rank of each individual
    ranks = [0 for _ in range(len_population)]
    
    for p in range(len_population):
        for q in range(len_population):
            # If p dominates q
            if non_dominance_condition(p, q, objectives):
                dominated[p].append(q)
            # If q dominates p
            elif non_dominance_condition(q, p, objectives):
                dominating[p] += 1
                
        # If p is dominated by no one, add it to the first front
        if dominating[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)
    
    # next fronts
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            # For each individual dominated by p, remove p from their dominators
            for q in dominated[p]:
                dominating[q] -= 1
                # If q has no other dominator, add it to the next front
                if dominating[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        # if its over, then next_front is empty and while loop stops
        fronts.append(next_front)
    return ranks, fronts[0:-1] # remove last empty front

def crowding_distance(front: list, objectives: list) -> list:
    """
    Calculate the crowding distance of each individual in the front.
    The crowding distance is a measure of how close an individual is to its neighbors in the objective space.
    """
    distance = [0.0 for _ in front]
    num_objectives = len(objectives[0]) # = 2 here
    
    for i in range(num_objectives):
        # Sort the individuals in the front by the i-th objective
        sorted_indices = sorted(range(len(front)), key=lambda x: objectives[x][i])
        # Set the distance of the first and last individual to infinity
        distance[sorted_indices[0]] = float("inf")
        distance[sorted_indices[-1]] = float("inf")
        
        # Calculate the distance for each individual in the front
        for j in range(1, len(sorted_indices) - 1):
            if objectives[sorted_indices[j]][i] != objectives[sorted_indices[0]][i]:
                distance[sorted_indices[j]] += (objectives[sorted_indices[j + 1]][i] - objectives[sorted_indices[j - 1]][i]) / (objectives[sorted_indices[-1]][i] - objectives[sorted_indices[0]][i])
    return distance

def tournament_selection(population: list, objectives: list, ranks: list, fronts: list) -> str:
    """
    Select an individual from the population using tournament selection.
    """
    # Select two random individuals
    ind1, ind2 = random.sample(range(len(population)), 2)
    
    # Compare their ranks
    if ranks[ind1] < ranks[ind2]:
        return population[ind1]
    elif ranks[ind1] > ranks[ind2]:
        return population[ind2]
    
    # They have the same rank
    
    rank = ranks[ind1]
    front_index1 = fronts[rank].index(ind1)
    front_index2 = fronts[rank].index(ind2)
    distances = crowding_distance(fronts[rank], objectives)
    distance1 = distances[front_index1]
    distance2 = distances[front_index2]
    
    if distance1 > distance2:
        return population[ind1]
    else:
        return population[ind2]
    
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

def selection(objectives: list, fronts: list, amount: int) -> list:
    selected = []
    current_rank = 0
    while len(selected) < amount:
        if len(fronts[current_rank]) + len(selected) <= amount:
            selected.extend(fronts[current_rank])
            current_rank += 1
        else:
            distances = crowding_distance(fronts[current_rank], objectives)
            sorted_indices = sorted(range(len(fronts[current_rank])), key=lambda x: distances[x], reverse=True)
            selected.extend(sorted_indices[:amount - len(selected)])
            break
    return selected

def genetic_algorithm(lookup_dict, features_amount, population_size=100, generations=100, mutation_rate=0.01) -> list:
    population = [generate_individual(features_amount) for _ in range(population_size)]
    objectives = [(function1(individual, lookup_dict), function2(individual, lookup_dict)) for individual in population]
    ranks, fronts = fast_non_dominated_sort(objectives)
    saved_generations = []
    
    for _ in tqdm(range(generations)):
        parent1 = tournament_selection(population, objectives, ranks, fronts)
        parent2 = tournament_selection(population, objectives, ranks, fronts)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        population.append(child1)
        population.append(child2)
        # non-dominated sorting on intermediate population : parents + children
        ranks, fronts = fast_non_dominated_sort(objectives)
        # Select the next generation
        selected = selection(objectives, fronts, population_size)
        # Create the new population
        population = [population[i] for i in selected]
        # Saving the generation
        saved_generations.append(population)
    # Return the best individuals
    best_individuals = []
    for i in range(len(population)):
        if ranks[i] == 0:
            best_individuals.append(population[i])
    return best_individuals

def plot_population(population: list, lookup_dict: dict, name: str) -> None:
    """
    Plot the population in the objective space.
    """
    plt.close()
    f1 = [function1(individual, lookup_dict) for individual in population]
    f2 = [function2(individual, lookup_dict) for individual in population]
    plt.scatter(f1, f2)
    
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    plt.title("Population in Objective Space")
    plt.show()
    plt.savefig(f"outputs/{name}_nsga2_final_population.png")
    plt.close()
    
def plot_generations(generations: list, lookup_dict: dict, name: str) -> None:
    """
    Plot the generations in the objective space.
    """
    plt.close()
    for i, generation in enumerate(generations):
        f1 = [function1(individual, lookup_dict) for individual in generation]
        f2 = [function2(individual, lookup_dict) for individual in generation]
        plt.scatter(f1, f2, label=f"Generation {i}")
    
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    plt.title("Generations in Objective Space")
    plt.legend()
    plt.show()
    plt.savefig(f"outputs/{name}_nsga2_generations.png")
    plt.close()

def plot_fronts(population: list, fronts: list, lookup_dict: dict) -> None:
    """
    Plot the fronts of the population.
    """
    plt.close()
    for i, front in enumerate(fronts):
        f1 = [function1(population[individual], lookup_dict) for individual in front]
        f2 = [function2(population[individual], lookup_dict) for individual in front]
        plt.scatter(f1, f2, label=f"Front {i}")
    
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    plt.title("Pareto Fronts")
    plt.legend()
    plt.show()
    plt.savefig("outputs/pareto_fronts.png")
    plt.close()

if __name__ == "__main__":
    # Dataset features amount : glass = 9, wine = 13, magic = 10
    
    table = csv_to_dict("outputs/wine_complete_table.csv")
    features_amount = len(next(iter(table.keys())))
    
    pareto_front = genetic_algorithm(table, features_amount, population_size=100, generations=100,  mutation_rate=0.01)
    
    # Plot the population
    plot_population(pareto_front, table, "wine")