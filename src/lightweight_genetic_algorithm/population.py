from abc import ABC, abstractmethod
import numpy as np

class Gene(ABC):
    """
    The abstract base class for a gene.
    Each subclass will define a gene in some genotype space.

    Attributes
    ----------
    mutation_methods : list of str
        The mutation methods that can be applied to the gene.

    Methods
    -------
    random_initialization()
        Creates a gene with random value.
    set_value(value)
        Sets the value of the gene.
    """
    mutation_methods = []

    @abstractmethod
    def random_initialization(self):
        pass

    @abstractmethod
    def set_value(self):
        pass

class NumericGene(Gene):
    """
    A numeric gene is represented by a real number within a range.

    Parameters
    ----------
    gene_range : tuple
        The range of the gene values.
    value : float, optional
        The value of the gene. If not provided, the gene will be initialized randomly.

    Attributes
    ----------
    low : float
        The lower bound of the gene values.
    high : float
        The upper bound of the gene values.
    value : float
        The value of the gene.

    Methods
    -------
    get_gene_range()
        Returns the gene range.
    random_initialization()
        Creates a gene with random value.
    set_value(value)
        Sets the value of the gene.
    copy()
        Returns a copy of the gene.

    """
    mutation_methods = ["additive", "multiplicative", "random"]
    crossover_methods = ["between", "midpoint", "either or"]

    def __init__(self, gene_range, value=None):
        self.low, self.high = gene_range
        self.value = value if value is not None else self.random_initialization()

    def get_gene_range(self):
        return (self.low, self.high)
    
    def random_initialization(self):
        return np.random.uniform(low=self.low, high=self.high)
    
    def set_value(self, value):
        self.value = value
    
    def copy(self):
        return NumericGene((self.low, self.high), self.value)

class CategoricalGene(Gene):
    """
    A categorical gene can take any value from a set of categories.

    Parameters
    ----------
    categories : list
        The allowed categories for the gene.
    value : object, optional
        The value of the gene. If not provided, the gene will be initialized randomly.

    Attributes
    ----------
    categories : list
        The allowed categories for the gene.
    value : object
        The value of the gene.

    Methods
    -------
    random_initialization()
        Creates a gene with random value.
    set_value(value)
        Sets the value of the gene.
    copy()
        Returns a copy of the gene.
    """
    mutation_methods = ["categorical"]
    crossover_methods = ["either or"]

    def __init__(self, categories, value=None):
        self.categories = categories
        if value is not None and value not in self.categories:
            raise ValueError("A categorical gene is being set to a value not in the allowed categories.")
        self.value = value if value is not None else self.random_initialization()

    def random_initialization(self):
        return np.random.choice(self.categories)
    
    def set_value(self, value):
        if value not in self.categories:
            raise ValueError("A categorical gene is being set to a value not in the allowed categories.")
        else:
            self.value = value
    
    def copy(self):
        return CategoricalGene(self.categories, self.value)
    
class Individual:
    """
    Class representing an individual in the population. The individual is defined by its genes. 

    Parameters
    ----------
    genes : list of Gene objects
        The genes that define the individual.
    fitness_function : function
        The fitness function that is used to calculate the fitness of the individual. The function should take a list of gene values as first argument and return a scalar value.
    fitness_function_args : tuple
        Additional arguments for the fitness function.
    fitness : float, optional
        The fitness of the individual. If not provided, the fitness function will be evaluated. Provides a way to avoid redundant evaluations of the fitness function.

    Attributes
    ----------
    genes : list of Gene objects
        The genes that define the individual.
    genes_values : numpy array
        The values of the genes.
    fitness_function : function
        The fitness function that is used to calculate the fitness of the individual.
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
        Returns the fitness function.
    copy()
        Returns a copy of the individual.

    """
    def __init__(self, genes, fitness_function, fitness_function_args, fitness=None):
        self.genes = np.array( [gene.copy() for gene in genes] )
        self.genes_values = np.array([gene.value for gene in self.genes])
        self.fitness_function = fitness_function
        self.fitness_function_args = fitness_function_args
        if fitness is None:
            try:
                self.fitness = fitness_function(self.genes_values,*self.fitness_function_args)
            except:
                raise ValueError("Error in fitness function evaluation. Your fitness function does not seem to be compatible with your individuals.")
        else:
            self.fitness = fitness

    def get_genes(self):
        return np.array([gene.copy() for gene in self.genes])
    
    def get_gene_values(self):
        return self.genes_values.copy()
    
    def get_fitness_function(self):
        return self.fitness_function
    
    def copy(self):
        return Individual(self.get_genes(), self.fitness_function, self.fitness_function_args, fitness=self.fitness)

