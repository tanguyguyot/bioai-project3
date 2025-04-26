# PARTICLE SWARM OPTIMIZATION (binary)

import numpy as np
from table_creation_test import csv_to_dict
from tqdm import tqdm

class Particle:
    def __init__(self, feature_length: int, lookup_table: dict):
        self.position: np.array = generate_individual(feature_length)
        self.velocity: np.array = np.random.randn(feature_length)
        self.fitness: float = fitness(self.position, lookup_table=lookup_table)
        self.best_position: np.array = self.position.copy()
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
    fitness_value = lookup_table.get(bitstring, np.inf).get("Error", np.inf)
    return fitness_value

def get_velocity(particle: Particle, inertia: float, b: float, c: float, global_best: np.array, Vmax: float=4.0) -> list:
    """
    Calculate the velocity of a particle.
    """
    r1 = np.random.rand(particle.position.size)
    r2 = np.random.rand(particle.position.size)
    cognitive = b * r1 * (particle.best_position - particle.position)
    social    = c * r2 * (global_best - particle.position)
    velocity = inertia * particle.velocity + cognitive + social
    return np.clip(velocity, -Vmax, Vmax)

def find_best_individual(population: list, lookup_table: dict) -> tuple:
    """
    Find the best individual amongst the population
    """
    best_individual = min(population, key=lambda p: p.fitness)
    best_position = best_individual.position
    best_fitness = best_individual.fitness
    return (best_position, best_fitness)

def update_particle(particle: Particle) -> Particle:
    """
    Update the particle's position and fitness based on calculated velocities.
    """
    probs = 1 / (1 + np.exp(-particle.velocity))
    # tirage vectorisé
    rand = np.random.rand(particle.position.size)
    particle.position = (rand < probs).astype(int)
    return particle
    
def particle_swarm_optimization(lookup_table: dict, num_particles: int = 50, num_iterations: int = 50, b: float = 1.5, c: float = 1.5, verbose=False) -> str:
    """
    Perform Particle Swarm Optimization.
    """
    feature_length = len(next(iter(lookup_table)))
    # Initialize particles : they already have fitness values so no need to re-evaluate
    particles = [Particle(feature_length, lookup_table) for _ in range(num_particles)]
    
    global_best_position, global_best_fitness = find_best_individual(particles, lookup_table)
    
    for generation in tqdm(range(num_iterations)):
        # change inertia value over time
        inertia = 1 - (0.6 * generation / num_iterations)
        
        # First loop to evaluate fitness and update best values
        for particle in particles:
            # Evaluate fitness of particle's current position
            particle.fitness = fitness(particle.position, lookup_table)
            # Check if better than personal best
            if particle.fitness < particle.best_fitness:
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness
            # Compare with global best
            if particle.fitness < global_best_fitness:
                global_best_position = particle.position.copy()
                global_best_fitness = particle.fitness
                
        # Second loop to update velocities and positions, because global best can change in first loop
        for particle in particles:
            # 4) update vélocité
            particle.velocity = get_velocity(particle, inertia, b, c, global_best_position)
            # 5) update position
            update_particle(particle)
        if generation % 10 == 0 and verbose:
            print(f"Gen {generation}, best fitness = {global_best_fitness}")
    print("Final best fitness: ", global_best_fitness)
    print("Final best position: ", global_best_position)
    return global_best_position, global_best_fitness

if __name__ == '__main__':
    # Dataset features amount : glass = 9, wine = 13, magic = 10
    # glass_lookup_dict = csv_to_dict("outputs/glass_complete_table.csv")
    # wine_lookup_dict = csv_to_dict("outputs/wine_complete_table.csv")
    # magic_lookup_dict = csv_to_dict("outputs/magi c_complete_table.csv")
    
    # zoo_lookup_dict = csv_to_dict("outputs/zoo_complete_table.csv")
    # heart_lookup_dict = csv_to_dict("outputs/heart_diseases_complete_table.csv")
    letters_lookup_dict = csv_to_dict("outputs/letters_complete_table.csv")
    
    # Test datasets
    # wine_best_position, wine_best_fitness = particle_swarm_optimization(wine_lookup_dict, num_particles=10, num_iterations=100, b=1.5, c=1.5)
    # glass_best_position, glass_best_fitness = particle_swarm_optimization(glass_lookup_dict, num_particles=10, num_iterations=100, b=1.5, c=1.5)
    # magic_best_position, magic_best_fitness = particle_swarm_optimization(magic_lookup_dict, num_particles=10, num_iterations=100, b=1.5, c=1.5)
    # zoo_best_position, zoo_best_fitness = particle_swarm_optimization(zoo_lookup_dict, num_particles=10, num_iterations=100, b=1.5, c=1.5)
    # heart_best_position, heart_best_fitness = particle_swarm_optimization(heart_lookup_dict, num_particles=10, num_iterations=100, b=1.5, c=1.5)
    letters_best_position, letters_best_fitness = particle_swarm_optimization(letters_lookup_dict, num_particles=50, num_iterations=100, b=1.5, c=1.5)
    
    # print("Best position for wine dataset: ", wine_best_position, " with fitness: ", wine_best_fitness)
    # print("Best position for glass dataset: ", glass_best_position, " with fitness: ", glass_best_fitness)
    # print("Best position for magic dataset: ", magic_best_position, " with fitness: ", magic_best_fitness)
    # print("Best position for zoo dataset: ", zoo_best_position, " with fitness: ", zoo_best_fitness)
    # print("Best position for heart dataset: ", heart_best_position, " with fitness: ", heart_best_fitness)
    print("Best position for letters dataset: ", letters_best_position, " with fitness: ", letters_best_fitness)