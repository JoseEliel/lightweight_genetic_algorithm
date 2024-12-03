import numpy as np
from .population import Individual

class Mutation:
    """
    A class used to represent mutations in a genetic algorithm.

    The Mutation class provides methods to mutate genes or individuals based on specified
    mutation modes and probabilities.

    Attributes:
        mutation_modes (list of str): List of mutation modes for each gene.
        mutation_probability (float): Probability of mutating each gene.
        param_ranges (list of tuple): List of parameter ranges for each gene.
        is_categorical (bool): Indicates if the genes are categorical.
    """

    def __init__(self, mutation_modes, mutation_probability, param_ranges):
        """
        Initializes the Mutation object with mutation modes, probability, and parameter ranges.

        Args:
            mutation_modes (list of str): A list of mutation modes for each gene.
            mutation_probability (float): The probability of mutating each gene. If None, defaults to 1/len(param_ranges).
            param_ranges (list of tuple): A list of parameter ranges (min, max) for each gene.

        Raises:
            ValueError: If mutation_probability is not between 0 and 1.
        """
        if mutation_probability is not None and not (0 <= mutation_probability <= 1):
            raise ValueError("mutation_probability must be between 0 and 1.")
        
        self.mutation_modes = mutation_modes
        self.mutation_probability = mutation_probability if mutation_probability else 1.0 / len(param_ranges)
        self.param_ranges = param_ranges
        self.is_categorical = len(param_ranges) != len(mutation_modes)

    def mutate_genes(self, genes, force_mutate=False):
        """
        Mutates a list of genes based on the mutation probability and modes.

        Args:
            genes (list of Gene): A list of Gene objects to mutate.
            force_mutate (bool, optional): If True, ensures at least one gene is mutated.

        Returns:
            list of Gene: The mutated list of Gene objects.

        Raises:
            ValueError: If a mutation mode is not compatible with a gene type.
        """
        # Choose which genes to mutate
        genes_to_mutate = [np.random.rand() < self.mutation_probability for _ in range(len(genes))]

        # If no gene was chosen to mutate, force the mutation of one gene (unless force_mutate is False)
        if not any(genes_to_mutate) and force_mutate:
            genes_to_mutate[np.random.randint(len(genes))] = True

        for i, gene in enumerate(genes):
            if genes_to_mutate[i]:
                if self.mutation_modes[i] not in gene.mutation_methods:
                    raise ValueError(
                        f"The mutation mode '{self.mutation_modes[i]}' is not compatible with the gene type."
                    )
                # Call the appropriate mutation method
                if self.is_categorical:
                    self.categorical(gene)
                else:
                    mutation_method = getattr(self, self.mutation_modes[i])
                    mutation_method(gene, self.param_ranges[i])

        return genes

    def mutate_individual(self, individual, force_mutate=False):
        """
        Mutates an individual by mutating its genes.

        Args:
            individual (Individual): The Individual object to mutate.
            force_mutate (bool, optional): If True, ensures at least one gene is mutated.

        Returns:
            Individual: A new Individual object with mutated genes.

        Raises:
            TypeError: If the input is not an instance of Individual.
        """
        if not isinstance(individual, Individual):
            raise TypeError("The mutate_individual method expects an instance of Individual.")
        
        mutated_genes = self.mutate_genes(individual.get_genes(), force_mutate)
        mutated_individual = Individual(
            mutated_genes,
            individual.get_fitness_function(),
            individual.fitness_function_args
        )
        return mutated_individual

    def additive(self, gene, param_range):
        """
        Applies an additive mutation to a gene.

        The gene's value is adjusted by adding a random value drawn from a normal distribution.

        Args:
            gene (Gene): The gene to mutate.
            param_range (tuple): The (min, max) range of the gene's parameter.

        Returns:
            Gene: The mutated gene.
        """
        range_size = abs(param_range[1] - param_range[0])
        std_dev = range_size / 10  # Standard deviation for mutation
        mutation_value = np.random.normal(loc=0.0, scale=std_dev)
        gene.set_value(gene.value + mutation_value)

    def multiplicative(self, gene, param_range=None):
        """
        Applies a multiplicative mutation to a gene.

        The gene's value is adjusted by multiplying it by a random factor drawn from a normal distribution centered at 1.

        Args:
            gene (Gene): The gene to mutate.
            param_range (tuple, optional): Not used in this method.

        Returns:
            Gene: The mutated gene.
        """
        mutation_factor = np.random.normal(loc=1, scale=0.5)
        gene.set_value(gene.value * mutation_factor)

    def random(self, gene, param_range):
        """
        Applies either an additive or multiplicative mutation to a gene at random.

        Args:
            gene (Gene): The gene to mutate.
            param_range (tuple): The (min, max) range of the gene's parameter.

        Returns:
            Gene: The mutated gene.
        """
        if np.random.rand() < 0.5:
            self.multiplicative(gene, param_range)
        else:
            self.additive(gene, param_range)

    def categorical(self, gene):
        """
        Mutates a categorical gene by randomly reinitializing its value.

        Args:
            gene (Gene): The categorical gene to mutate.

        Returns:
            Gene: The mutated gene.
        """
        gene.set_value(gene.random_initialization())