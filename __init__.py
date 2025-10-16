"""
lattice_ga package initializer.

This file makes the core classes and functions from the modules available at the package level,
simplifying imports for the user. For example, you can use:
    from lattice_ga import Lattice, crossover, mutation, GeneticAlgorithm
"""
from .lattice import Lattice
from .operators import crossover, mutation
from .ga_core import GeneticAlgorithm
