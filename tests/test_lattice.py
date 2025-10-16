#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:48:45 2025

@author: sanup
"""

import numpy as np
import pytest
import sys
import os


# --- IDE-Friendly Package Importing ---
try:
    from lattice_ga.lattice import Lattice
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(project_root, 'src'))
    from lattice_ga.lattice import Lattice


# A valid unimodular integer basis for testing
VALID_BASIS = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])


def test_lattice_creation_success():
    """Tests that a Lattice object is created successfully with a valid basis."""
    lattice = Lattice(VALID_BASIS)
    assert np.array_equal(lattice.basis, VALID_BASIS)
    assert lattice.dim == 3

    # Check if the Gram matrix is calculated correctly
    expected_gram = VALID_BASIS.T @ VALID_BASIS
    assert np.array_equal(lattice.gram_matrix, expected_gram)


def test_fails_on_non_square_matrix():
    """Tests that Lattice creation fails if the matrix is not square."""
    non_square = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="must be a square"):
        Lattice(non_square)


def test_fails_on_non_integer_matrix():
    """Tests that Lattice creation fails if the matrix contains non-integers."""
    non_integer = np.array([[1.5, 2], [3, 4]])
    with pytest.raises(ValueError, match="must be integers"):
        Lattice(non_integer)


def test_fails_on_non_unimodular_matrix():
    """Tests that Lattice creation fails if the matrix determinant is not +/- 1."""
    # This matrix has a determinant of 2
    non_unimodular = np.array([[1, 1], [0, 2]])
    with pytest.raises(ValueError, match="must be unimodular"):
        Lattice(non_unimodular)
