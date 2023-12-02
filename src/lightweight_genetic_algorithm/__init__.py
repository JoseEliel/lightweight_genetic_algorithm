"""
The lightweight_genetic_algorithm package
An intuitive, flexible and efficient implementation of a genetic algorithm in Python
"""

# Version of the simple_genetic_algorithm package
__version__ = "0.1"

from .algorithm import GeneticAlgorithm
from .crossover import CrossoverBetween, CrossoverMidpoint, CrossoverEitherOr
from .selection import SurvivorSelection, DiversityEnhancedSurvivorSelection, FitnessProportionalSurvivorSelection
#from .diversity import Diversity
