#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:41:43 2025

@author: sanup
"""

import numpy as np


# class Lattice:
#     """
#     Represents a lattice defined by a basis matrix.

#     This class performs essential pre-computation checks upon initialization to ensure
#     the lattice is valid for the isometry problem with Z^n. Specifically, it verifies
#     that the basis matrix contains only integers and is unimodular.

#     Attributes:
#         basis (np.ndarray): The integer basis matrix of the lattice.
#         gram_matrix (np.ndarray): The Gram matrix (B^T * B) of the lattice.
#         dim (int): The dimension of the lattice (n for an n x n basis).
#     """

#     def __init__(self, basis_matrix):
#         """
#         Initializes the Lattice object and performs validation.

#         Args:
#             basis_matrix (array-like): A list of lists or a NumPy array representing
#                                        the basis matrix of the lattice.

#         Raises:
#             ValueError: If the basis matrix is not composed of integers,
#                         is not square, or is not unimodular (i.e., |det(B)| != 1).
#         """
#         # --- 1. Convert to NumPy array and check shape ---
#         matrix = np.array(basis_matrix, dtype=np.float64)

#         if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
#             raise ValueError("Basis matrix must be a square 2D array.")

#         self.dim = matrix.shape[0]

#         # --- 2. Integer Entry Check ---
#         if not np.allclose(matrix, np.round(matrix)):
#             raise ValueError(
#                 "All entries in the basis matrix must be integers.")

#         self.basis = matrix.astype(int)

#         # --- 3. Unimodular Check ---
#         det = np.linalg.det(self.basis)
#         if not np.isclose(abs(det), 1.0):
#             raise ValueError(
#                 f"Basis matrix must be unimodular (|det(B)| = 1), but got |det(B)| = {abs(det):.4f}."
#             )

#         # --- 4. Compute and store the Gram Matrix ---
#         self.gram_matrix = self.basis.T @ self.basis

#     def __repr__(self):
#         """Provides a developer-friendly representation of the Lattice."""
#         return f"Lattice(dim={self.dim}, basis=\n{self.basis}\n)"


class Lattice:
    """
    Represents a lattice's geometric structure via its Gram matrix.

    This class performs essential pre-computation checks upon initialization to ensure
    the input matrix is a valid Gram matrix for a unimodular integer lattice.
    Specifically, it verifies that the matrix is symmetric, positive-definite,
    composed of integers, and has a determinant of 1.

    Attributes:
        gram_matrix (np.ndarray): The integer Gram matrix of the lattice.
        dim (int): The dimension of the lattice.
    """

    def __init__(self, gram_matrix):
        """
        Initializes the Lattice object and validates the input Gram matrix.

        Args:
            gram_matrix (array-like): A list of lists or a NumPy array representing
                                      the Gram matrix of the lattice.

        Raises:
            ValueError: If the Gram matrix is not integer, square, symmetric,
                        positive-definite, or does not have a determinant of 1.
        """
        # --- 1. Convert to NumPy array and check shape ---
        matrix = np.array(gram_matrix, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Gram matrix must be a square 2D array.")

        self.dim = matrix.shape[0]

        # --- 2. Symmetry Check ---
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Gram matrix must be symmetric.")

        # --- 3. Integer Entry Check ---
        if not np.allclose(matrix, np.round(matrix)):
            raise ValueError(
                "All entries in the Gram matrix must be integers.")

        # --- 4. Positive-Definite Check ---
        # A valid Gram matrix must be positive-definite (all eigenvalues > 0).
        eigenvalues = np.linalg.eigvalsh(matrix)
        if not np.all(eigenvalues > 0):
            raise ValueError("Gram matrix must be positive-definite.")

        # If all checks pass, we can safely convert to an integer type.
        self.gram_matrix = matrix.astype(int)

        # --- 5. Unimodular Check (for Gram Matrix) ---
        # For a Gram matrix G = B.T * B, det(G) = det(B)^2.
        # If the lattice is unimodular (|det(B)|=1), then det(G) must be 1.
        det = np.linalg.det(self.gram_matrix)
        if not np.isclose(det, 1.0):
            raise ValueError(
                f"Gram matrix determinant must be 1 for a unimodular lattice, but got {det:.4f}."
            )

    def __repr__(self):
        """Provides a developer-friendly representation of the Lattice."""
        return f"Lattice(dim={self.dim}, gram_matrix=\n{self.gram_matrix}\n)"
