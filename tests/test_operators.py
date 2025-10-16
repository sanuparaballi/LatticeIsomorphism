#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:50:18 2025

@author: sanup
"""

import numpy as np
import pytest
import sys
import os

# --- IDE-Friendly Package Importing ---
try:
    from lattice_ga.operators import crossover, mutation
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(project_root, 'src'))
    from lattice_ga.operators import crossover, mutation


# Define two valid 3x3 unimodular integer matrices for testing
PARENT_1 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
PARENT_2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # det = 1


def test_crossover_preserves_shape_and_type():
    """Tests that crossover produces a matrix of the same shape and integer type."""
    child = crossover(PARENT_1, PARENT_2)
    assert child.shape == PARENT_1.shape
    assert child.dtype == int


def test_crossover_preserves_unimodular_property():
    """Tests that the product of two unimodular matrices is also unimodular."""
    # det(A*B) = det(A) * det(B). So, 1 * 1 = 1.
    child = crossover(PARENT_1, PARENT_2)
    det_child = np.linalg.det(child)
    assert np.isclose(abs(det_child), 1.0)


def test_mutation_preserves_shape_and_type():
    """Tests that mutation produces a matrix of the same shape and integer type."""
    mutated = mutation(PARENT_1.copy())
    assert mutated.shape == PARENT_1.shape
    assert mutated.dtype == int


def test_mutation_preserves_unimodular_property():
    """
    Tests that mutation by an elementary matrix preserves the unimodular property.
    The determinant of the elementary matrix used is 1.
    """
    mutated = mutation(PARENT_1.copy())
    det_mutated = np.linalg.det(mutated)
    assert np.isclose(abs(det_mutated), 1.0)


def test_mutation_changes_matrix():
    """Tests that mutation actually alters the matrix (for n > 1)."""
    # Run a few times to reduce the chance of a random no-op
    for _ in range(10):
        mutated = mutation(PARENT_1.copy())
        if not np.array_equal(mutated, PARENT_1):
            assert True
            return
    # If it never changed after 10 tries, something is likely wrong
    assert False, "Mutation failed to change the matrix over 10 attempts."
