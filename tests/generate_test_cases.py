#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 18:21:00 2025

@author: sanup
"""

import numpy as np
import os

try:
    from lattice_ga.operators import mutation
except ImportError:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(project_root, 'src'))
    from lattice_ga.operators import mutation


def generate_positive_gram_matrix(dim: int, scramble_steps: int) -> np.ndarray:
    """Generates a valid Gram matrix for Z^n."""
    print(f"Generating {dim}D positive test case (Gram matrix)...")
    basis = np.identity(dim, dtype=int)
    for _ in range(scramble_steps):
        basis = mutation(basis, mutation_strength=1)

    # Compute and return the Gram matrix
    gram_matrix = basis.T @ basis
    print("Positive case generated.")
    return gram_matrix


def generate_negative_gram_matrix(dim: int) -> np.ndarray:
    """Generates a Gram matrix for a lattice not isometric to Z^n."""
    print(f"Generating {dim}D negative test case (Gram matrix)...")
    basis = np.identity(dim, dtype=int)
    basis[0, 0] = 2  # This makes the basis non-unimodular

    # Compute and return the Gram matrix. Its determinant will not be 1.
    gram_matrix = basis.T @ basis
    assert not np.isclose(np.linalg.det(gram_matrix), 1.0)
    print("Negative case generated.")
    return gram_matrix


def main():
    """Main function to generate and save test case Gram matrices."""
    DIMENSION = 256

    positive_gram = generate_positive_gram_matrix(
        dim=DIMENSION, scramble_steps=DIMENSION * 5)
    negative_gram = generate_negative_gram_matrix(dim=DIMENSION)

    output_path = os.path.join(os.path.dirname(
        __file__), f"test_data_gram_{DIMENSION}d.npz")

    # Save both Gram matrices to a single compressed .npz file
    np.savez_compressed(
        output_path,
        positive_gram=positive_gram,
        negative_gram=negative_gram
    )

    print(f"\nTest case Gram matrices successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
