import numpy as np

class Diversity:
    """
    A class to calculate the diversity score between two individuals.

    ...

    Attributes
    ----------
    B0 : float
        The diversity constant.
    measure : function
        The function to measure the distance between two points. Default is Euclidean distance.

    Methods
    -------
    set_population_size(population_size):
        Sets the population size and calculates the diversity constant B0.
    compute_diversity(point, survivor):
        Calculates the diversity score between an individual and a survivor using the measure function and the diversity constant B0.
    """

    def __init__(self, measure):
        self.B0 = None
        self.measure = measure if measure else lambda point_a, point_b: np.linalg.norm(point_a - point_b)

    def set_population_size(self, population_size):
        self.B0 = 1 / population_size

    def compute_diversity(self, point, survivor):
        r = self.measure(point,survivor)
        r0 = 0.5
        diversity_result = self.B0 * np.exp(-r / r0)
        return diversity_result