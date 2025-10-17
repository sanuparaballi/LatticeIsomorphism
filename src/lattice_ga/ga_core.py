#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:45:41 2025

@author: sanup
"""

import numpy as np
import random
from typing import List, Tuple
from scipy.linalg import cholesky
from fpylll import IntegerMatrix, LLL

from .operators import crossover, error_guided_mutation


class GeneticAlgorithm:
    """
    The core engine for the genetic algorithm search.

    This class manages a population of candidate unimodular matrices (U) and
    evolves them over generations to find a matrix that minimizes the fitness
    function || G - U^T * U ||_F.

    The updated core engine for the Intelligent Evolution Strategy.

    This class implements the full memetic algorithm, including:
    1. LLL-informed population initialization.
    2. Evolution using crossover and error-guided mutation.
    3. Local Search Intensification (polishing) of elite solutions.
    """

    def __init__(self, target_gram_matrix: np.ndarray, population_size: int,
                 n_generations: int, mutation_rate: float, n_elites: int,
                 local_search_steps: int = 5):
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
        self.local_search_steps = local_search_steps  # New parameter
        self.population: List[np.ndarray] = []
        self.best_solution: Tuple[np.ndarray, float] = (np.array([]), np.inf)

    def _initialize_population(self):
        """
        Creates an initial population seeded by an LLL-reduced basis.

        This provides a high-quality starting point for the search.
        """
        print("Performing Cholesky + LLL reduction for initial seed...")
        # 1. Get a real-valued basis from G via Cholesky decomposition
        # G = L * L^T. We use B = L^T as our initial real basis.
        try:
            l_matrix = cholesky(self.target_g, lower=True)
            real_basis = l_matrix.T
        except np.linalg.LinAlgError:
            # If G is not positive-definite, fall back to identity
            real_basis = np.identity(self.dim)

        # 2. Convert to an fpylll IntegerMatrix for LLL reduction
        # We need to scale the real basis to be integer for LLL.
        scale_factor = 1e5
        scaled_numpy_matrix = (real_basis * scale_factor).astype(np.int64)

        # --- FIX: Convert NumPy array to a list of lists for fpylll ---
        scaled_list_of_lists = scaled_numpy_matrix.tolist()
        int_basis = IntegerMatrix.from_matrix(scaled_list_of_lists)

        # 3. Run LLL reduction
        lll_reduced_basis = LLL.reduction(int_basis)

        # Convert back to a NumPy array and rescale.
        # This gives us a "nice" real-valued basis.
        lll_as_list = [list(row) for row in lll_reduced_basis]
        seed_matrix_real = np.array(
            lll_as_list, dtype=np.float64) / scale_factor

        # Finally, round to the nearest integer matrix to get our seed.
        # This is our best initial guess for a unimodular basis.
        seed_matrix = np.round(seed_matrix_real).astype(int)

        # Ensure the seed is unimodular, if not, fallback to identity
        if not np.isclose(abs(np.linalg.det(seed_matrix)), 1.0):
            seed_matrix = np.identity(self.dim, dtype=int)

        # --- Verify LLL seed and fallback if necessary ---
        seed_fitness = self._calculate_fitness(seed_matrix)
        SEED_FITNESS_THRESHOLD = 5000.0

        if seed_fitness > SEED_FITNESS_THRESHOLD:
            print("\n" + "="*50)
            print("WARNING: LLL reduction produced a low-quality initial seed.")
            print(
                f"Seed Fitness: {seed_fitness:.2f} (High is bad). This can be due to library versions.")
            print("FALLING BACK to the identity matrix as the seed.")
            print("="*50 + "\n")
            # Fallback to the safe default
            seed_matrix = np.identity(self.dim, dtype=int)

        print("Seed generated. Initializing population...")
        self.population = [seed_matrix]
        while len(self.population) < self.population_size:
            # Create diverse individuals by slightly mutating the high-quality seed
            new_ind = error_guided_mutation(seed_matrix.copy(), self.target_g)
            self.population.append(new_ind)

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
            fitness_scores = [self._calculate_fitness(
                u) for u in self.population]
            sorted_population = sorted(
                zip(self.population, fitness_scores), key=lambda x: x[1])
            self.population = [ind for ind, score in sorted_population]

            if sorted_population[0][1] < self.best_solution[1]:
                self.best_solution = sorted_population[0]

            print(
                f"Generation {generation+1}/{self.n_generations} | Best Fitness: {self.best_solution[1]:.6f}")

            if np.isclose(self.best_solution[1], 0.0):
                print("Perfect solution found!")
                return self.best_solution

            next_generation = []
            elites = self.population[:self.n_elites]

            # --- NEW: Local Search Intensification ---
            polished_elites = []
            for elite in elites:
                temp_elite = elite
                for _ in range(self.local_search_steps):
                    temp_elite = error_guided_mutation(
                        temp_elite, self.target_g)
                polished_elites.append(temp_elite)

            next_generation.extend(polished_elites)

            # --- Crossover & Mutation ---
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.choices(
                    self.population[:self.population_size // 2], k=2)
                child = crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    # Use the new intelligent mutation
                    child = error_guided_mutation(child, self.target_g)

                next_generation.append(child)

            self.population = next_generation

        print("GA run finished.")
        return self.best_solution
