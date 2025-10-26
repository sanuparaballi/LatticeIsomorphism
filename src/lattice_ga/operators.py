#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:44:03 2025

@author: sanup
"""

import numpy as np
import random


def crossover(parent1: np.ndarray, parent2: np.ndarray, target_g: np.ndarray, repair_steps: int = 5) -> np.ndarray:
    """
    Performs crossover by multiplying two parent matrices.

    Since the product of two matrices in GL(n, Z) is also in GL(n, Z),
    this operator ensures the offspring remains within the valid search space.

    Args:
        parent1 (np.ndarray): The first parent matrix (U1).
        parent2 (np.ndarray): The second parent matrix (U2).

    Returns:
        np.ndarray: The new child matrix (U_new = U1 * U2).
    """
    # return parent1 @ parent2

    n = parent1.shape[0]
    child = np.zeros_like(parent1)

    # Uniform crossover: for each column, pick one from either parent.
    for j in range(n):
        if random.random() < 0.5:
            child[:, j] = parent1[:, j]
        else:
            child[:, j] = parent2[:, j]

    # Repair the newly formed child to be closer to a valid solution
    repaired_child = child
    for _ in range(repair_steps):
        repaired_child = error_guided_mutation(repaired_child, target_g)

    return repaired_child


# def error_guided_mutation(u_matrix: np.ndarray, target_g: np.ndarray, mutation_strength: int = 1) -> np.ndarray:
#     """
#     Applies an intelligent mutation guided by the fitness error.

#     This operator identifies the largest error in the current solution's Gram
#     matrix (G - U^T*U) and applies a targeted elementary transformation to
#     the two columns of U that are causing that error. This is far more
#     efficient than a random mutation.

#     Args:
#         u_matrix (np.ndarray): The matrix to be mutated.
#         target_g (np.ndarray): The target Gram matrix (G).
#         mutation_strength (int): The maximum absolute value of the integer 'r'
#                                  used in the elementary matrix. Defaults to 1.

#     Returns:
#         np.ndarray: The intelligently mutated matrix.
#     """
#     n = u_matrix.shape[0]
#     if n < 2:
#         return u_matrix

#     # 1. Calculate the Error Matrix: E = G - U^T * U
#     current_g = u_matrix.T @ u_matrix
#     error_matrix = target_g - current_g

#     # 2. Find the Biggest Mistake
#     # We set the diagonal to zero to focus on off-diagonal elements, which
#     # represent the incorrect angles between basis vectors.
#     np.fill_diagonal(error_matrix, 0)

#     # Find the indices (i, j) of the element with the largest absolute value.
#     # This tells us which two columns are most in need of correction.
#     i, j = np.unravel_index(
#         np.argmax(np.abs(error_matrix)), error_matrix.shape)

#     if i == j:  # Should not happen with diagonal zeroed, but as a safeguard
#         return u_matrix

#     # 3. Apply a Targeted Fix
#     # Choose a small integer 'r' for the transformation.
#     # A simple choice is to use the sign of the error to guide the direction,
#     # though a random small integer also works well.
#     r = random.choice([-mutation_strength, mutation_strength])
#     if error_matrix[i, j] > 0:
#         r = abs(r)
#     else:
#         r = -abs(r)

#     # 4. Create the elementary matrix E and apply the mutation
#     elementary_matrix = np.identity(n, dtype=int)
#     elementary_matrix[i, j] = r

#     mutated_u = u_matrix @ elementary_matrix

#     return mutated_u


def _calculate_fitness_for_op(u_matrix, target_g):
    """Helper function to calculate fitness directly."""
    diff = target_g - (u_matrix.T @ u_matrix)
    return np.linalg.norm(diff, 'fro')


def error_guided_mutation(u_matrix: np.ndarray, target_g: np.ndarray, mutation_strength: int = 3) -> np.ndarray:
    """
    Applies a powerful "Greedy Search" mutation. (EA v3.3)

    This version tests a small set of step sizes 'r' and greedily chooses
    the one that results in the largest fitness improvement.
    """
    n = u_matrix.shape[0]
    if n < 2:
        return u_matrix

    current_g = u_matrix.T @ u_matrix
    error_matrix = target_g - current_g
    np.fill_diagonal(error_matrix, 0)

    i, j = np.unravel_index(
        np.argmax(np.abs(error_matrix)), error_matrix.shape)
    if i == j:
        return u_matrix

    best_mutation = u_matrix
    best_fitness = _calculate_fitness_for_op(u_matrix, target_g)

    # --- Greedy Search for the best 'r' ---
    # Test a range of small integer steps to find the most effective one.
    possible_rs = list(range(-mutation_strength, 0)) + \
        list(range(1, mutation_strength + 1))

    for r in possible_rs:
        temp_matrix = u_matrix.copy()
        elementary_matrix = np.identity(n, dtype=np.int64)
        elementary_matrix[i, j] = r

        mutated_u = temp_matrix @ elementary_matrix
        current_fitness = _calculate_fitness_for_op(mutated_u, target_g)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_mutation = mutated_u

    return best_mutation
