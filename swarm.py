# PARTICLE SWARM OPTIMIZATION (binary)

import random
import numpy as np
from table_creation import csv_to_dict
from tqdm import tqdm

class Particle:
    def __init__(self, feature_length: int, lookup_table: dict):
        self.position: np.array = generate_individual(feature_length)
        self.velocity: np.array = np.random.randn(feature_length)
        self.fitness: float = fitness(self.position, lookup_table=lookup_table)
        self.best_position: np.array = self.position
        self.best_fitness: float = self.fitness
        
def sigmoid(velocitities: np.array) -> np.array:
    """
    Apply sigmoid function to convert velocities to probabilities.
    """
    return 1 / (1 + np.exp(-velocitities))
        
def generate_individual(length: int) -> np.array:
    """
    Generate a random individual with length genes.
    """
    individual: np.array = np.random.randint(0, 2, length)
    return individual

def fitness(individual: np.array, lookup_table: dict) -> int:
    """
    Calculate the fitness of an individual based on the lookup table.
    """
    bitstring = ''.join(str(int(gene)) for gene in individual)
    fitness_value = lookup_table.get(bitstring, np.inf).get("Lookup value", np.inf)
    return fitness_value

def get_velocity(particle: Particle, inertia: float, b: float, c: float, global_best: np.array) -> list:
    """
    Calculate the velocity of a particle.
    """
    R = random.uniform(0, 1)
    return inertia * particle.velocity + b * R * (particle.best_position - particle.position) + c * R * (global_best - particle.position)

def find_best_individual(population: list, lookup_table: dict) -> Particle:
    """
    Find the best individual amongst the population
    """
    best_individual = min(population, key=lambda p: p.fitness)
    best_fitness = best_individual.fitness
    return best_individual, best_fitness

def update_particle(particle: Particle) -> Particle:
    """
    Update the particle's position and fitness based on calculated velocities.
    """
    velocities = particle.velocity
    probabilities = sigmoid(velocities)
    for i in range(len(particle.position)):
        if random.random() < probabilities[i]:
            particle.position[i] = 1
        else:
            particle.position[i] = 0
    return particle
    
def particle_swarm_optimization(lookup_table: dict, num_particles: int, num_iterations: int, inertia: float, b: float, c: float) -> str:
    """
    Perform Particle Swarm Optimization.
    """
    feature_length = len(next(iter(lookup_table)))
    # Initialize particles : they already have fitness values so no need to re-evaluate
    particles = [Particle(feature_length, lookup_table) for _ in range(num_particles)]
    
    global_best_position, global_best_fitness = find_best_individual(particles, lookup_table)
    
    for _ in tqdm(range(num_iterations)):
        for particle in particles:
            # Evaluate fitness of particle's current position
            particle.fitness = fitness(particle.position, lookup_table)
            # Check if better than personal best
            if particle.fitness < particle.best_fitness:
                particle.best_position = particle.position
                particle.best_fitness = particle.fitness
            # Compare with global best
            if particle.fitness < global_best_fitness:
                global_best_position = particle.position
                global_best_fitness = particle.fitness
            # Update velocity
            particle.velocity = get_velocity(particle, inertia, b, c, global_best_position)
            # Update position based on probabilities
            particle = update_particle(particle, lookup_table)
        print("Current best fitness: ", global_best_fitness)
    print("Final best fitness: ", global_best_fitness)
    print("Final best position: ", global_best_position)
    return global_best_position, global_best_fitness

if __name__ == '__main__':
    # Dataset features amount : glass = 9, wine = 13, magic = 10
    glass_lookup_dict = csv_to_dict("outputs/glass_complete_table.csv")
    wine_lookup_dict = csv_to_dict("outputs/wine_complete_table.csv")
    magic_lookup_dict = csv_to_dict("outputs/magic_complete_table.csv")
    
    best_position, best_fitness = particle_swarm_optimization(wine_lookup_dict, num_particles=100, num_iterations=1000, inertia=0.5, b=1.5, c=1.5)