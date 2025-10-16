#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:47:14 2025

@author: sanup
"""

# from lattice_ga import Lattice, GeneticAlgorithm
import numpy as np
import sys
import os

# --- Add the source directory to the Python path ---
# This allows us to import our 'lattice_ga' package from the 'src' directory.
# It's a common pattern for making research project scripts runnable.
# The code below is a fallback for running the script directly without installation.
try:
    from lattice_ga.lattice import Lattice
    from lattice_ga.ga_core import GeneticAlgorithm
except ImportError:
    # Append the source directory to the path if the package is not installed
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(project_root, 'src'))
    from lattice_ga.lattice import Lattice
    from lattice_ga.ga_core import GeneticAlgorithm


# ===================================================================
# Main Experiment Setup
# ===================================================================


def main():
    """Defines and runs a single GA experiment."""

    # --- 1. Define the Input Lattice Basis (B) ---
    # This is a simple unimodular matrix that is known to be a basis for Z^3.
    # It's the identity matrix with column 0 added to column 1.
    # Our GA should be able to find a U such that G = U^T * U.
    basis_b = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    print("--- Experiment Setup ---")
    print(f"Input Lattice Basis (B):\n{basis_b}\n")

    # --- 2. Create Lattice Object and Perform Pre-checks ---
    try:
        lattice = Lattice(basis_matrix=basis_b)
        print(f"Target Gram Matrix (G = B^T * B):\n{lattice.gram_matrix}\n")
    except ValueError as e:
        print(f"Error: Lattice validation failed. {e}")
        return

    # --- 3. Set Genetic Algorithm Hyperparameters ---
    ga_params = {
        'population_size': 100,
        'n_generations': 50,
        'mutation_rate': 0.2,
        'n_elites': 5
    }
    print("GA Hyperparameters:")
    for key, val in ga_params.items():
        print(f"  - {key}: {val}")
    print("------------------------\n")

    # --- 4. Initialize and Run the GA ---
    ga = GeneticAlgorithm(
        target_gram_matrix=lattice.gram_matrix,
        **ga_params
    )
    best_u, best_fitness = ga.run()

    # --- 5. Display Results ---
    print("\n--- Results ---")
    print(f"Final Fitness Score (d): {best_fitness:.6f}")
    print(f"Best Solution Matrix (U) found:\n{best_u}")

    if np.isclose(best_fitness, 0.0):
        print("\nConclusion: The lattice IS LIKELY isometric to Z^n.")
    else:
        print("\nConclusion: The lattice IS NOT isometric to Z^n based on this run.")


if __name__ == "__main__":
    main()
