import numpy as np

class Diversity:
    """
    A class to calculate the diversity score between two individuals.

    The diversity score is a measure of how different two individuals
    are from each other, based on their parameter values.

    Default is Euclidean. 

    If "paper" is given as a measure, then the measure from arXiv:XXX.XXX is used.

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

    def __init__(self, measure):
        self.B0 = None
        # If measure is 'paper', set measure function to measure_paper
        # If measure is None or not provided, set measure function to measure_euclidean
        # If some other function is provided, use that as measure function
        if measure == 'paper':
            self.measure = self.measure_paper
        else:
            self.measure = measure if measure else self.measure_euclidean

    # Static method for calculating Euclidean distance
    @staticmethod
    def measure_euclidean(point_a, point_b):
        return np.sum((point_a - point_b)**2)
        #return sum((x - y) ** 2 for x, y in zip(point_a, point_b))

    # Static method for calculating 'paper' measure
    @staticmethod
    def measure_paper(point, survivor):
        epsilon = 1e-10
        r = np.sum((point - survivor)**2 / (np.abs(point) + np.abs(survivor) + epsilon)**2)
        return r

    # Set population size and calculate the diversity constant B0
    def set_population_size(self, population_size):
        self.B0 = 1 / population_size

    # Compute diversity between an individual and a survivor
    # Use the measure method set in the constructor and the B0 constant
    def compute_diversity(self, individual, survivor):
        # Extract gene values from individuals
        point = individual.get_gene_values()
        survivor_point = survivor.get_gene_values()

        # Check if genes are categorical or numeric
        if individual.get_genes()[0].__class__.__name__ == 'CategoricalGene':
            # Use Hamming distance for categorical genes
            distance_sq = np.sum(point != survivor_point) / len(point)
        else:
            # Use the measure method set in the constructor for numeric genes
            distance_sq = self.measure(point, survivor_point)

        r0 = 0.5
        diversity_result = self.B0 * np.exp(-distance_sq / r0)
        return diversity_result