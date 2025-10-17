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
import argparse

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


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================================================================
# Main Experiment Setup
# ===================================================================


# def main():
#     """Defines and runs a single GA experiment."""

#     # --- 1. Define the Input Gram Matrix (G) ---
#     # This is the Gram matrix for a simple unimodular basis of Z^3.
#     # It is pre-calculated from a basis B = [[1, 1, 0], [0, 1, 0], [0, 0, 1]].
#     gram_g = np.array([
#         [1, 1, 0],
#         [1, 2, 0],
#         [0, 0, 1]
#     ], dtype=int)

#     print("--- Experiment Setup ---")
#     print(f"Input Gram Matrix (G):\n{gram_g}\n")

#     # --- 2. Create Lattice Object and Perform Pre-checks ---
#     try:
#         # The Lattice class now takes the Gram matrix directly
#         lattice = Lattice(gram_matrix=gram_g)
#         print(f"Target Gram Matrix successfully validated.\n")
#     except ValueError as e:
#         print(f"Error: Gram matrix validation failed. {e}")
#         return

#     # --- 3. Set Genetic Algorithm Hyperparameters ---
#     ga_params = {
#         'population_size': 100,
#         'n_generations': 50,
#         'mutation_rate': 0.2,
#         'n_elites': 5
#     }
#     print("GA Hyperparameters:")
#     for key, val in ga_params.items():
#         print(f"  - {key}: {val}")
#     print("------------------------\n")

#     # --- 4. Initialize and Run the GA ---
#     ga = GeneticAlgorithm(
#         target_gram_matrix=lattice.gram_matrix,
#         **ga_params
#     )
#     best_u, best_fitness = ga.run()

#     # --- 5. Display Results ---
#     print("\n--- Results ---")
#     print(f"Final Fitness Score (d): {best_fitness:.6f}")
#     print(f"Best Solution Matrix (U) found:\n{best_u}")

#     if np.isclose(best_fitness, 0.0):
#         print("\nConclusion: The lattice IS LIKELY isometric to Z^n.")
#     else:
#         print("\nConclusion: The lattice IS NOT isometric to Z^n based on this run.")


# if __name__ == "__main__":
#     main()


def main():
    """
    Loads a high-dimensional test case from a .npz file and runs the GA.
    """
    # --- 1. Set up Argument Parser ---
    np.random.seed(42)
    # random.seed(42)

    parser = argparse.ArgumentParser(
        description="Run GA for Lattice Isometry.")
    parser.add_argument(
        '--case',
        type=str,
        choices=['positive', 'negative'],
        help="Which test case to run ('positive' or 'negative')."
    )

    # --- Configuration for Spyder/IDEs ---
    # If running from the command line, it will use your provided arguments.
    # If you run this script directly in an IDE (like pressing F5 in Spyder),
    # it will use the default arguments specified in 'ide_args'.
    # CHANGE THE VALUE BELOW TO 'negative' TO RUN THE NEGATIVE TEST CASE.
    ide_default_case = 'positive'

    if len(sys.argv) <= 1:
        # No command-line arguments provided, use IDE defaults
        args = parser.parse_args([f'--case={ide_default_case}'])
    else:
        # Command-line arguments were provided, use them
        args = parser.parse_args()

    # --- 2. Load the specified test case ---
    dim = 256
    test_data_filename = f"test_data_gram_{dim}d.npz"
    # The data file is in the 'tests' directory, one level up from 'experiments'
    test_data_path = os.path.join(project_root, 'tests', test_data_filename)

    try:
        data = np.load(test_data_path)
        if args.case == 'positive':
            gram_g = data['positive_gram']
        else:  # negative case
            gram_g = data['negative_gram']
    except FileNotFoundError:
        print(f"Error: Test data file not found at '{test_data_path}'.")
        print("Please run 'python tests/generate_test_cases.py' first.")
        return

    # --- DIAGNOSTIC: Check the loaded data type ---
    print(f"DEBUG: Loaded matrix data type is {gram_g.dtype}")

    print("--- Experiment Setup ---")
    print(f"Running {dim}D '{args.case}' test case.")
    print(f"Loaded Gram matrix from: {test_data_path}\n")

    # --- 3. Create Lattice Object and Perform Pre-checks ---
    try:
        lattice = Lattice(gram_matrix=gram_g)
        print(f"Target Gram Matrix successfully validated.\n")
    except ValueError as e:
        print(f"Error: Gram matrix validation failed. {e}")
        return

    # --- 4. Set GA Hyperparameters for a large problem ---
    # NOTE: A 256-dimensional search space is vast. These parameters
    # may need significant tuning to find a solution.
    ga_params = {
        'population_size': 200,    # Increased for larger search space
        'n_generations': 1000,   # Increased for more evolution time
        'mutation_rate': 0.3,
        'n_elites': 10             # Keep more of the best solutions
    }
    print("GA Hyperparameters:")
    for key, val in ga_params.items():
        print(f"  - {key}: {val}")
    print("------------------------\n")

    # --- 5. Initialize and Run the GA ---
    ga = GeneticAlgorithm(
        target_gram_matrix=lattice.gram_matrix,
        **ga_params
    )
    best_u, best_fitness = ga.run()

    # --- 6. Display Results ---
    print("\n--- Results ---")
    print(f"Final Fitness Score (d): {best_fitness:.6f}")
    # We won't print the 256x256 matrix, just its fitness.
    print(f"Best Solution Matrix (U) found has shape: {best_u.shape}")

    if np.isclose(best_fitness, 0.0):
        print("\nConclusion: The lattice IS LIKELY isometric to Z^n.")
    else:
        print("\nConclusion: Based on this run, the lattice does not appear to be isometric to Z^n.")


if __name__ == "__main__":
    main()
