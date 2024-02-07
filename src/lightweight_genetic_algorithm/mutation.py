import numpy as np
from .population import Individual

class Mutation:
    """
    A class used to represent a Mutation
    """
    def __init__(self, mutation_modes, mutation_probability, param_ranges):
        self.mutation_modes = mutation_modes
        self.mutation_probability = mutation_probability if mutation_probability else 1.0/len(param_ranges)
        self.param_ranges = param_ranges
        self.is_categorical = len(param_ranges) != len(mutation_modes)

    def mutate_genes(self, genes, force_mutate=False):
        '''
        Mutate a list of genes. Give a list of Gene objects as input, return a list of Gene objects.
        '''
        # Choose which genes to mutate
        genes_to_mutate = [ True if np.random.rand() < self.mutation_probability else False for i in range(len(genes))]
        
        # If no gene was chosen to mutate, force the mutation of one gene (unless force_mutate is False)
        if np.sum(genes_to_mutate) == 0 and force_mutate:
            genes_to_mutate[np.random.randint(len(genes))] = True

        for i, gene in enumerate(genes):
            # Check if a mutation should occur based on the specified mutation rate.
            if genes_to_mutate[i]:
                # Check if the mutation mode is compatible with the gene
                if self.mutation_modes[i] not in gene.mutation_methods:
                    raise ValueError(f"The mutation mode '{self.mutation_modes[i]}' is not compatible with the gene type.")
                # Call the mutation method and update the gene
                if self.is_categorical:
                    gene = getattr(self, self.mutation_modes[i])(gene, None)
                else:
                    #print("gene ranges from ", self.param_ranges[i])
                    gene = getattr(self, self.mutation_modes[i])(gene, self.param_ranges[i])
                    #print("to ", self.param_ranges[i])

        
        return genes

    def mutate_individual(self, individual, force_mutate=False):
        '''
        Mutate an individual by mutating its genes. Give an Individual as input, return a mutated Individual.
        '''
        if not isinstance(individual, Individual):
            raise TypeError("The mutate method expects an instance of Individual.")
        
        mutated_genes = self.mutate_genes(individual.get_genes(), force_mutate)
        mutated_individual = Individual(mutated_genes,individual.get_fitness_function(),individual.fitness_function_args)
        return mutated_individual
    
    def additive(self, gene, param_range):
        # Calcualtes 
        
        range_size = abs(param_range[1] - param_range[0])
        lowest = -range_size/2
        highest = range_size/2

        # gene.set_value(gene.value + np.random.uniform(low=lowest, high=highest))
        # same as above but with gaussian
        gene.set_value(gene.value + np.random.normal(loc=0.0, scale= (highest-lowest)/10, size = None))
        return gene

    def multiplicative(self, gene, param_range = None):
        gene.set_value(gene.value * np.random.normal(loc=1, scale=0.5))
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