#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:44:03 2025

@author: sanup
"""

import numpy as np
import random


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
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
    return parent1 @ parent2


def mutation(chromosome_u: np.ndarray, mutation_strength: int = 1) -> np.ndarray:
    """
    Applies a mutation by multiplying with a random elementary matrix.

    An elementary matrix E(i, j, r) is an identity matrix with one additional
    off-diagonal entry, E_ij = r. Multiplying by E on the right performs a
    column operation (adds r * column_i to column_j), which is a valid
    unimodular transformation for integer r.

    Args:
        chromosome_u (np.ndarray): The matrix to be mutated.
        mutation_strength (int): The maximum absolute value of the integer 'r'
                                 used in the elementary matrix. Defaults to 1.

    Returns:
        np.ndarray: The mutated matrix.
    """
    n = chromosome_u.shape[0]
    if n < 2:
        return chromosome_u  # Cannot mutate a 1x1 matrix this way

    # 1. Choose two different indices for the column operation
    i, j = random.sample(range(n), 2)

    # 2. Choose a small integer 'r' for the transformation
    # It must be non-zero.
    r = random.choice(
        list(range(-mutation_strength, 0)) +
        list(range(1, mutation_strength + 1))
    )

    # 3. Create the elementary matrix E
    elementary_matrix = np.identity(n, dtype=int)
    elementary_matrix[i, j] = r

    # 4. Apply the mutation (a column operation)
    mutated_u = chromosome_u @ elementary_matrix

    return mutated_u
