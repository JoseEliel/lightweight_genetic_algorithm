import numpy as np

class Diversity:
    """
    A class to calculate the diversity score between two individuals.

    The diversity score is a measure of how different two individuals
    are from each other, based on their parameter values.

    Default is Euclidean. 

    If "dynamic" is given as a measure, then the measure from arXiv:XXX.XXX is used.

    The class also includes a method to adjust the diversity calculation based
    on the size of the total population. 
    
    ...

    Attributes
    ----------
    measure : function
        The function used to measure the distance between two individuals.
   
    Methods
    -------
    __init__(self, measure):
        Initializes the Diversity object with a specific measure function.
    set_population_size(self, population_size):
        Sets the population size and calculates the diversity constant B0.
    compute_diversity(self, point, survivor):
        Compute diversity between an individual and a survivor.
    """ 

    def __init__(self, measure, B0=1, r0=0.5):
        self.B0 = B0  # Diversity punishment for identical individuals
        self.r0 = r0  # Characteristic distance for the diversity measure
        self.measure = measure # Measure function to evaluate distances between individuals

    # Set population size and calculate the diversity constant B0
    def set_population_size(self, population_size):
        self.B0 = 1 / population_size

    # Compute diversity between an individual and a survivor
    # Use the measure method set in the constructor and the B0 constant
    def compute_diversity(self, individual, survivor):

        # Extract gene values from individuals
        point = individual.get_gene_values()
        survivor_point = survivor.get_gene_values()

        distance_sq = self.measure(point, survivor_point)

        diversity_result = self.B0 * np.exp(-distance_sq / self.r0)
        return diversity_result