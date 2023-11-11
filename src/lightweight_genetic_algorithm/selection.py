import numpy as np

# Base class for survivor selection methods
class SurvivorSelection:
    """
    A base class for survivor selection methods.

    This class is used to define the interface for survivor selection methods.
    The class is not intended to be used directly, but rather to be inherited
    by other classes that define specific survivor selection methods.

    ...
    
    Attributes
    ----------
    None
    
    Methods
    -------
    select_survivors(self, population, surviving_population_size):
        Selects the survivors from a population. Returns a list of individuals of size surviving_population_size.
    """ 

    # Select survivors from a population
    def select_survivors(self, population, surviving_population_size):
        """
        Selects the survivors from a population. Returns a list of individuals of size surviving_population_size.

        Parameters
        ----------
        population : list
            A list of individuals.
        surviving_population_size : int
            The number of individuals to select from the population.

        Returns
        -------
        list
            A list of individuals of size surviving_population_size.
        """ 
        raise NotImplementedError("select_survivors() method not implemented.")
    

# Diversity enhanced survivor selection
class DiversityEnhancedSurvivorSelection(SurvivorSelection):
    """
    A class for diversity enhanced survivor selection.

    This class implements the diversity enhanced survivor selection method.
    The method selects the survivors from the population based both the fitness
    and the diversity of the individuals.

    ...

    Attributes
    ----------
    r0 : float
        The characteristic distance beyond which there is no diversity punishment.
    B0 : float
        Diversity punishment for identical individuals.
    measure : function
        The function used to measure the "distance" or "dissimilarity" between two individuals.

    Methods
    -------
    select_survivors(self, population, surviving_population_size):
        Selects the survivors from a population. Returns a list of individuals of size surviving_population_size.
    
    compute_diversity(self, point, survivor):
        Compute diversity punishment for an individual (point) given a survivor.
    """ 

    def __init__(self, measure, r0=.707, B0=1.0):
        """
        Constructs all the necessary attributes for the diversity enhanced survivor selection method.

        Parameters
        ----------
        measure : function or string
            The function used to measure the "distance" or "dissimilarity" between two individuals.
        r0 : float
            The characteristic distance beyond which there is no diversity punishment.
        B0 : float
            Diversity punishment for identical individuals.

        """

        self.r0 = r0
        self.B0 = B0

        # If the measure is a string, then use the corresponding measure function
        if measure == "euclidean":
            self.measure = lambda x,y: np.sum((x - y)**2)   
        elif measure == "hamming":
            self.measure = lambda x,y: np.sum(x != y) / len(x) 
        elif measure == "dynamic":
            self.measure = lambda x,y: np.sum((x - y)**2 / (np.abs(x) + np.abs(y) + 1e-10)**2)
        else:
            self.measure = measure

    # Compute diversity between an individual and a survivor
    def compute_diversity(self, individual, survivor):

        # Extract gene values from individuals
        point = individual.get_gene_values()
        survivor_point = survivor.get_gene_values()

        distance_sq = self.measure(point, survivor_point)

        diversity_result = self.B0 * np.exp(-distance_sq / self.r0)
        return diversity_result

    # Select survivors from a population
    def select_survivors(self, population, surviving_population_size, B0=None):
        """
        Selects the survivors from a population. Returns a list of individuals of size surviving_population_size.

        Parameters
        ----------
        population : list
            A list of individuals.
        surviving_population_size : int
            The number of individuals to select from the population.
        B0 : float (optional)
            Diversity punishment for identical individuals.
        r0 : float (optional)
            The characteristic distance beyond which there is no diversity punishment.

        Returns
        -------
        list
            A list of individuals of size surviving_population_size.
        """ 
        
        # Set self.B0 to 1 / population size if not given
        if B0 is None:
            self.B0 = 1. / len(population)
        else:
            self.B0 = B0

        # List to keep selected survivors
        survivors = []

        population = np.array(population)
        diversity_scores = [individual.fitness for individual in population]

        for i in range(surviving_population_size):
            # Get the best survivor and remove it from population
            best_survivor_idx = np.argmax(diversity_scores)
            survivors.append(population[best_survivor_idx])
            population = np.delete(population, best_survivor_idx)
            diversity_scores = np.delete(diversity_scores, best_survivor_idx)
        
            # Update the diversity score of remaining individuals
            for i,individual in enumerate(population):
                diversity_punishment = self.compute_diversity(individual, survivors[-1])
                diversity_scores[i] -= diversity_punishment

        return survivors
