from abc import ABC, abstractmethod
import numpy as np

class Gene(ABC):
    """
    Abstract base class for a gene.

    Each subclass defines a gene in a specific genotype space.

    Attributes
    ----------
    mutation_methods : list of str
        The mutation methods that can be applied to the gene.

    Methods
    -------
    random_initialization()
        Provides a random value appropriate for the gene.
    set_value()
        Sets the value of the gene.
    """
    mutation_methods = []

    @abstractmethod
    def random_initialization(self):
        """
        Provides a random value appropriate for the gene.

        Returns
        -------
        value
            A random value suitable for initializing the gene.
        """
        pass

    @abstractmethod
    def set_value(self):
        """
        Sets the value of the gene.

        Note
        ----
        Implementation should define how the value is set.
        """
        pass

class NumericGene(Gene):
    """
    A numeric gene represented by a real number within a range.

    Parameters
    ----------
    gene_range : tuple of float
        The range (low, high) of the gene values.
    value : float, optional
        The value of the gene. If not provided, the gene will be initialized with a random value.

    Attributes
    ----------
    low : float
        The lower bound of the gene values.
    high : float
        The upper bound of the gene values.
    value : float
        The current value of the gene.

    Methods
    -------
    get_gene_range()
        Returns the gene range as a tuple (low, high).
    random_initialization()
        Generates and returns a random value within the gene range.
    set_value(value)
        Sets the value of the gene to the specified value.
    copy()
        Creates and returns a copy of the gene.
    """

    mutation_methods = ["additive", "multiplicative", "random"]
    crossover_methods = ["between", "midpoint", "either or"]

    def __init__(self, gene_range, value=None):
        self.low, self.high = gene_range
        self.value = value if value is not None else self.random_initialization()

    def get_gene_range(self):
        """
        Returns the gene range.

        Returns
        -------
        tuple of float
            A tuple (low, high) representing the gene range.
        """
        return (self.low, self.high)
    
    def random_initialization(self):
        """
        Generates and returns a random value within the gene range.

        Returns
        -------
        float
            A random value within the gene range.
        """
        return np.random.uniform(low=self.low, high=self.high)
    
    def set_value(self, value):
        """
        Sets the value of the gene to the specified value.

        Parameters
        ----------
        value : float
            The new value for the gene.
        """
        self.value = value
    
    def copy(self):
        """
        Creates and returns a copy of the gene.

        Returns
        -------
        NumericGene
            A new instance of NumericGene with the same range and value.
        """
        return NumericGene((self.low, self.high), self.value)

class CategoricalGene(Gene):
    """
    A categorical gene that can take any value from a set of categories.

    Parameters
    ----------
    categories : list
        The allowed categories for the gene.
    value : object, optional
        The value of the gene. Must be one of the allowed categories. If not provided, the gene will be initialized with a random value.

    Attributes
    ----------
    categories : list
        The allowed categories for the gene.
    value : object
        The current value of the gene.

    Methods
    -------
    random_initialization()
        Selects and returns a random value from the categories.
    set_value(value)
        Sets the value of the gene to the specified value.
    copy()
        Creates and returns a copy of the gene.

    Raises
    ------
    ValueError
        If the provided `value` is not in the allowed categories.
    """

    mutation_methods = ["categorical"]
    crossover_methods = ["either or"]

    def __init__(self, categories, value=None):
        self.categories = categories
        if value is not None and value not in self.categories:
            raise ValueError("A categorical gene is being set to a value not in the allowed categories.")
        self.value = value if value is not None else self.random_initialization()

    def random_initialization(self):
        """
        Selects and returns a random value from the categories.

        Returns
        -------
        object
            A random value from the allowed categories.
        """
        return np.random.choice(self.categories)
    
    def set_value(self, value):
        """
        Sets the value of the gene to the specified value.

        Parameters
        ----------
        value : object
            The new value for the gene.

        Raises
        ------
        ValueError
            If the provided `value` is not in the allowed categories.
        """
        if value not in self.categories:
            raise ValueError("A categorical gene is being set to a value not in the allowed categories.")
        else:
            self.value = value
    
    def copy(self):
        """
        Creates and returns a copy of the gene.

        Returns
        -------
        CategoricalGene
            A new instance of CategoricalGene with the same categories and value.
        """
        return CategoricalGene(self.categories, self.value)
    
class Individual:
    """
    Represents an individual in the population, defined by its genes.

    Parameters
    ----------
    genes : list of Gene
        The genes that define the individual.
    fitness_function : callable
        The fitness function used to calculate the fitness of the individual.
        The function should take a list of gene values as its first argument and return a scalar value.
    fitness_function_args : tuple
        Additional arguments for the fitness function.
    fitness : float, optional
        The fitness of the individual. If not provided, the fitness function will be evaluated.
        This allows avoiding redundant evaluations of the fitness function.

    Attributes
    ----------
    genes : numpy.ndarray
        An array containing the genes of the individual.
    genes_values : numpy.ndarray
        An array containing the values of the genes.
    fitness_function : callable
        The fitness function used to calculate the fitness of the individual.
    fitness_function_args : tuple
        Additional arguments for the fitness function.
    fitness : float
        The fitness of the individual.

    Methods
    -------
    get_genes()
        Returns a copy of the genes.
    get_gene_values()
        Returns a copy of the gene values.
    get_fitness_function()
        Returns the fitness function used by the individual.
    copy()
        Creates and returns a copy of the individual.

    Raises
    ------
    ValueError
        If the fitness function evaluation fails, indicating incompatibility with the individual's genes.
    """

    def __init__(self, genes, fitness_function, fitness_function_args, fitness=None):
        self.genes = np.array( [gene.copy() for gene in genes] )
        self.genes_values = np.array([gene.value for gene in self.genes])
        self.fitness_function = fitness_function
        self.fitness_function_args = fitness_function_args
        if fitness is None:
            try:
                self.fitness = fitness_function(self.genes_values, *self.fitness_function_args)
            except Exception:
                raise ValueError("Error in fitness function evaluation. Your fitness function does not seem to be compatible with your individuals.")
        else:
            self.fitness = fitness

    def get_genes(self):
        """
        Returns a copy of the genes.

        Returns
        -------
        numpy.ndarray
            An array containing copies of the individual's genes.
        """
        return np.array([gene.copy() for gene in self.genes])
    
    def get_gene_values(self):
        """
        Returns a copy of the gene values.

        Returns
        -------
        numpy.ndarray
            An array containing the values of the individual's genes.
        """
        return self.genes_values.copy()
    
    def get_fitness_function(self):
        """
        Returns the fitness function used by the individual.

        Returns
        -------
        callable
            The fitness function.
        """
        return self.fitness_function
    
    def copy(self):
        """
        Creates and returns a copy of the individual.

        Returns
        -------
        Individual
            A new Individual instance with the same genes and fitness.
        """
        return Individual(self.get_genes(), self.fitness_function, self.fitness_function_args, fitness=self.fitness)
