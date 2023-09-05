import numpy as np
from .population import Individual


class Mutation:
    """
    A class used to represent a Mutation
    """
    def __init__(self, mutation_modes, mutation_rate, param_ranges):
        self.mutation_modes = mutation_modes
        self.mutation_rate = mutation_rate if mutation_rate else 1.0/len(param_ranges)
        self.param_ranges = param_ranges
        self.is_categorical = len(param_ranges) != len(mutation_modes)

    def mutate(self, individual):
        if not isinstance(individual, Individual):
            raise TypeError("The mutate method expects an instance of Individual.")
        
        mutated_genes = []
        for i, gene in enumerate(individual.get_genes()):
            # Check if a mutation should occur based on the specified mutation rate.
            if np.random.rand() < self.mutation_rate:
                # Check if the mutation mode is compatible with the gene
                if self.mutation_modes[i] not in gene.mutation_methods:
                    raise ValueError(f"The mutation mode '{self.mutation_modes[i]}' is not compatible with the gene type.")
                # Call the mutation method
                if self.is_categorical:
                    mutated_gene = getattr(self, self.mutation_modes[i])(gene, None)
                else:
                    mutated_gene = getattr(self, self.mutation_modes[i])(gene, self.param_ranges[i])

                mutated_genes.append(mutated_gene)
            else:
                mutated_genes.append(gene)
        mutated_individual = Individual(mutated_genes,individual.get_fitness_function(),individual.fitness_function_args)
        return mutated_individual
    
    def additive(self, gene, param_range):
        gene.set_value(gene.value + np.random.uniform(low=param_range[0], high=param_range[1]))
        return gene

    def multiplicative(self, gene, param_range = None):
        gene.set_value(gene.value * np.random.uniform(low=-2, high=2))
        return gene

    def random(self, gene, param_range):
        # Randomly choose between additive and multiplicative mutation.
        if np.random.rand() < 0.5:
            self.multiplicative(gene, param_range)
        else:
            self.additive(gene, param_range)

    def categorical(self, gene, param_range = None):
        gene.set_value(gene.random_initialization())
        return gene