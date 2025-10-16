#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:45:41 2025

@author: sanup
"""

import numpy as np
import random
from typing import List, Tuple

from .operators import crossover, mutation


class GeneticAlgorithm:
    """
    The core engine for the genetic algorithm search.

    This class manages a population of candidate unimodular matrices (U) and
    evolves them over generations to find a matrix that minimizes the fitness
    function || G - U^T * U ||_F.
    """

    def __init__(self, target_gram_matrix: np.ndarray, population_size: int,
                 n_generations: int, mutation_rate: float, n_elites: int):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            target_gram_matrix (np.ndarray): The target Gram matrix (G).
            population_size (int): The number of individuals in the population.
            n_generations (int): The number of generations to run the evolution.
            mutation_rate (float): The probability of an individual being mutated.
            n_elites (int): The number of top individuals to carry over to the
                            next generation without change (elitism).
        """
        self.target_g = target_gram_matrix
        self.dim = self.target_g.shape[0]
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.n_elites = n_elites
        self.population: List[np.ndarray] = []
        self.best_solution: Tuple[np.ndarray, float] = (np.array([]), np.inf)

    def _initialize_population(self):
        """Creates the initial population of unimodular matrices."""
        self.population = []
        # Start with a simple population: identity matrices and simple mutations
        # This provides a stable, valid starting point.
        identity = np.identity(self.dim, dtype=int)
        self.population.append(identity)

        while len(self.population) < self.population_size:
            # Create diverse individuals by mutating the identity matrix
            mutated_id = mutation(identity.copy(), mutation_strength=1)
            self.population.append(mutated_id)

    def _calculate_fitness(self, u_matrix: np.ndarray) -> float:
        """Calculates the Frobenius norm distance: || G - U^T * U ||_F."""
        diff = self.target_g - (u_matrix.T @ u_matrix)
        return np.linalg.norm(diff, 'fro')

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Executes the main evolutionary loop.

        Returns:
            A tuple containing the best matrix U found and its final fitness score.
        """
        self._initialize_population()

        for generation in range(self.n_generations):
            # Calculate fitness for the entire population
            fitness_scores = [self._calculate_fitness(
                u) for u in self.population]

            # Sort population by fitness (lower is better)
            sorted_population = sorted(
                zip(self.population, fitness_scores), key=lambda x: x[1])
            self.population = [ind for ind, score in sorted_population]

            # Update the best solution found so far
            if sorted_population[0][1] < self.best_solution[1]:
                self.best_solution = sorted_population[0]

            print(
                f"Generation {generation+1}/{self.n_generations} | Best Fitness: {self.best_solution[1]:.4f}")

            # Check for perfect solution
            if np.isclose(self.best_solution[1], 0.0):
                print("Perfect solution found!")
                return self.best_solution

            # Create the next generation
            next_generation = []

            # 1. Elitism: Carry over the best individuals
            next_generation.extend(self.population[:self.n_elites])

            # 2. Crossover & Mutation
            while len(next_generation) < self.population_size:
                # Select two parents from the best half of the population
                parent1, parent2 = random.choices(
                    self.population[:self.population_size // 2], k=2)

                # Create child
                child = crossover(parent1, parent2)

                # Apply mutation
                if random.random() < self.mutation_rate:
                    child = mutation(child)

                next_generation.append(child)

            self.population = next_generation

        print("GA run finished.")
        return self.best_solution
