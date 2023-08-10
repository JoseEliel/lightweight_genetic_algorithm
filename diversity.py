import numpy as np

class Diversity:
    """
    A class to calculate the diversity score between two individuals.

    ...

    Attributes
    ----------
    number_of_parameters : int
        The number of parameters for each individual.

    Methods
    -------
    compute_diversity(point, survivor, population_size):
        Calculates the diversity score between an individual and a survivor.
    """
    def __init__(self, number_of_parameters):
        self.number_of_parameters = number_of_parameters

    def compute_diversity(self, point, survivor, population_size):
        r = sum((point[i] - survivor[i]) ** 2 for i in range(self.number_of_parameters))
        B0=1/population_size
        r0=0.5
        diversity_result = B0 * np.exp(-r/r0)

        return diversity_result