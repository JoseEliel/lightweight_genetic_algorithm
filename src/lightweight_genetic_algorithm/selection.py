import numpy as np

# Base class for survivor selection methods
class SurvivorSelection:
    """
    Base class for survivor selection methods in evolutionary algorithms.

    This abstract class defines the interface for survivor selection strategies.
    Subclasses should implement the `select_survivors` method to specify
    how survivors are selected from a population.
    """

    def select_survivors(self, population, surviving_population_size):
        """
        Select survivors from the population.

        Parameters
        ----------
        population : list
            A list of individuals in the current population.
        surviving_population_size : int
            The number of individuals to select as survivors.

        Returns
        -------
        list
            A list of selected individuals of length `surviving_population_size`.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("select_survivors() method not implemented.")


# Diversity enhanced survivor selection
class DiversityEnhancedSurvivorSelection(SurvivorSelection):
    """
    Diversity-enhanced survivor selection strategy.

    This class implements a survivor selection method that considers both
    the fitness and the diversity of individuals. It aims to maintain
    diversity in the population by penalizing individuals that are too
    similar to already selected survivors.

    Parameters
    ----------
    measure : callable or {'euclidean', 'hamming', 'dynamic'}
        A function to measure the distance or dissimilarity between two individuals.
        If a string is provided, it must be one of the predefined measures:
        - 'euclidean': Sum of squared differences (Euclidean distance squared).
        - 'hamming': Hamming distance normalized by the length of the vectors.
        - 'dynamic': Dynamic distance considering relative differences.
    r0 : float, optional
        The characteristic distance beyond which there is no diversity penalty.
        Default is 1.0.
    D0 : float, optional
        The maximum diversity penalty for identical individuals.
        Default is 1.0.

    Attributes
    ----------
    r0 : float
        Characteristic distance for diversity calculation.
    D0 : float
        Maximum diversity penalty.
    measure : callable
        Function to compute the distance between two individuals.
    """

    def __init__(self, measure, r0=None, D0=None):
        """
        Initialize the diversity-enhanced survivor selection strategy.

        Parameters
        ----------
        measure : callable or {'euclidean', 'hamming', 'dynamic'}
            Function or string specifying the distance measure.
        r0 : float, optional
            Characteristic distance for diversity calculation. Default is 1.0.
        D0 : float, optional
            Maximum diversity penalty. Default is 1.0.
        """
        if r0 is None:
            r0 = 1.0
        self.r0 = r0

        if D0 is None:
            D0 = 1.0
        self.D0 = D0

        # Define the measure function based on the input
        if measure == "euclidean":
            self.measure = lambda x, y: np.sum((x - y) ** 2)
        elif measure == "hamming":
            self.measure = lambda x, y: np.sum(x != y) / len(x)
        elif measure == "dynamic":
            self.measure = lambda x, y: np.sum((x - y) ** 2 / (np.abs(x) + np.abs(y) + 1e-10) ** 2)
        else:
            self.measure = measure

    def compute_diversity(self, individual, survivor):
        """
        Compute the diversity penalty between an individual and a selected survivor.

        The penalty decreases exponentially with the squared distance between
        the individual and the survivor.

        Parameters
        ----------
        individual : object
            An individual from the population. Must have a method `get_gene_values()`
            that returns a numpy array of gene values.
        survivor : object
            A survivor individual already selected. Must have a method `get_gene_values()`
            that returns a numpy array of gene values.

        Returns
        -------
        float
            The diversity penalty for the individual.
        """
        # Extract gene values from individuals
        point = individual.get_gene_values()
        survivor_point = survivor.get_gene_values()

        distance_sq = self.measure(point, survivor_point)

        diversity_penalty = self.D0 * np.exp(-distance_sq / self.r0 ** 2)
        return diversity_penalty

    def select_survivors(self, population, surviving_population_size):
        """
        Select survivors from the population based on fitness and diversity.

        The method iteratively selects the best individual (based on adjusted fitness),
        adds it to the list of survivors, and updates the fitness of the remaining
        individuals by subtracting the diversity penalty with respect to the newly
        added survivor.

        Parameters
        ----------
        population : list
            A list of individuals in the current population.
            Each individual must have a `fitness` attribute and a `get_gene_values()` method.
        surviving_population_size : int
            The number of individuals to select as survivors.

        Returns
        -------
        list
            A list of selected individuals of length `surviving_population_size`.

        Notes
        -----
        The method modifies the input `population` list by removing selected survivors.
        """
        # Initialize diversity-adjusted fitness scores
        adjusted_fitness = np.array([individual.fitness for individual in population])

        # List to keep selected survivors
        survivors = []

        for _ in range(surviving_population_size):
            # Select the individual with the highest adjusted fitness
            best_idx = np.argmax(adjusted_fitness)
            best_individual = population[best_idx]
            survivors.append(best_individual)

            # Remove the selected individual from consideration
            population.pop(best_idx)
            adjusted_fitness = np.delete(adjusted_fitness, best_idx)

            # Update diversity-adjusted fitness scores for remaining individuals
            for i, individual in enumerate(population):
                penalty = self.compute_diversity(individual, best_individual)
                adjusted_fitness[i] -= penalty

        # Sort survivors by original fitness in descending order
        survivors.sort(key=lambda ind: ind.fitness, reverse=True)

        return survivors


class FitnessProportionalSurvivorSelection(SurvivorSelection):
    """
    Fitness-proportional survivor selection strategy.

    This class implements a survivor selection method where individuals
    are selected based solely on their fitness values. Individuals with
    higher fitness have a higher chance of being selected.
    """

    def select_survivors(self, population, surviving_population_size):
        """
        Select survivors from the population based on fitness.

        Parameters
        ----------
        population : list
            A list of individuals in the current population.
            Each individual must have a `fitness` attribute.
        surviving_population_size : int
            The number of individuals to select as survivors.

        Returns
        -------
        list
            A list of selected individuals of length `surviving_population_size`.
        """
        # Sort individuals by fitness in descending order
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        # Select the top individuals
        survivors = sorted_population[:surviving_population_size]
        return survivors