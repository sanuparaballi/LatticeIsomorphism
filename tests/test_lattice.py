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

try:
    from lattice_ga import Lattice
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(project_root, 'src'))
    from lattice_ga import Lattice

# A valid Gram matrix for a unimodular lattice (Z^3)
VALID_GRAM_MATRIX = np.array([[1, 1, 0], [1, 2, 0], [0, 0, 1]])


def test_lattice_creation_success():
    """Tests that a Lattice object is created successfully with a valid Gram matrix."""
    lattice = Lattice(VALID_GRAM_MATRIX)
    assert np.array_equal(lattice.gram_matrix, VALID_GRAM_MATRIX)
    assert lattice.dim == 3


def test_fails_on_non_square_matrix():
    """Tests that Lattice creation fails if the matrix is not square."""
    non_square = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="must be a square"):
        Lattice(non_square)


def test_fails_on_non_integer_matrix():
    """Tests that Lattice creation fails if the matrix contains non-integers."""
    non_integer = np.array([[1.5, 2], [2, 4]])
    with pytest.raises(ValueError, match="must be integers"):
        Lattice(non_integer)


def test_fails_on_non_symmetric_matrix():
    """Tests that Lattice creation fails if the matrix is not symmetric."""
    non_symmetric = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="must be symmetric"):
        Lattice(non_symmetric)


def test_fails_on_non_positive_definite_matrix():
    """Tests that Lattice creation fails if the matrix is not positive-definite."""
    # This matrix has eigenvalues [3, -1], so it's not positive-definite
    not_pos_def = np.array([[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="must be positive-definite"):
        Lattice(not_pos_def)


def test_fails_on_non_unimodular_gram_matrix():
    """Tests that Lattice creation fails if the Gram matrix determinant is not 1."""
    # This matrix corresponds to a basis with det=2, so det(G) = 4
    non_unimodular_gram = np.array([[1, 0], [0, 4]])
    with pytest.raises(ValueError, match="determinant must be 1"):
        Lattice(non_unimodular_gram)
