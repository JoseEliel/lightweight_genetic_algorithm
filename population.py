from abc import ABC, abstractmethod
import numpy as np
class Gene(ABC):
    """
    The abstract base class for a gene.
    Each subclass will define a gene in some genotype space.
    """
    mutation_methods = []  # Add this line

    @abstractmethod
    def random_initialization(self):
        pass

    @abstractmethod
    def set_value(self):
        pass


class NumericGene(Gene):
    """
    A numeric gene is represented by a real number within a range.
    """
    mutation_methods = ["additive", "multiplicative", "random"]
    crossover_methods = ["between", "midpoint", "either or"]

    def __init__(self, low, high, value=None):
        self.low = low
        self.high = high
        self.value = value if value is not None else self.random_initialization()

    def random_initialization(self):
        return np.random.uniform(low=self.low, high=self.high)
    
    def set_value(self, value):
        self.value = value

class CategoricalGene(Gene):
    """
    A categorical gene can take any value in a given set.
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
    
class Individual:
    """
    An individual is defined by its genes.
    """
    def __init__(self, genes, fitness_function, fitness_function_args):
        self.genes = genes
        self.genes_values = np.array([gene.value for gene in genes])  # Add this line
        self.diversity_score = 0
        self.fitness_function = fitness_function
        self.fitness_function_args = fitness_function_args
        try:
            self.fitness = fitness_function(self.genes_values,*self.fitness_function_args)
        except:
            raise ValueError("Error in fitness function evaluation. Your fitness function does not seem to be compatible with your individuals.")

    def get_genes(self):
        return self.genes
    
    def get_gene_values(self):   # Add this method
        return self.genes_values
    
    def get_fitness_function(self):
        return self.fitness_function

    def set_fitness(self,value):
        self.fitness = value
    
    def set_diversity_score(self, value):
        self.diversity_score = value

