import numpy as np
from .diversity import Diversity
from .crossover import CrossoverBetween, CrossoverMidpoint, CrossoverEitherOr
from .mutation import Mutation
from .population import Individual, NumericGene, CategoricalGene
import warnings


class GeneticAlgorithm:
    """
    A class used to represent a Genetic Algorithm

    Attributes
    ----------
    fitness_function : function
        A function that calculates the fitness score of an individual
    gene_ranges : list
        A list of tuples representing the range of each parameter; if parameters are categorical, it's a simple 1D list of categories
    crossover_method : str
        The method used for crossover (default is "Either Or")
    number_of_genes : int
        The number of parameters (default is the length of gene_ranges). This must be explicitly provided if parameters are categorical.
    mutation_mode : list
        The mode used for mutation (default is "additive" for numeric parameters and "categorical" for categorical parameters)
    mutation_rate : float
        The rate of mutation (default is 1.0/number_of_genes)
    diversity : Diversity
        An instance of the Diversity class used to calculate diversity scores
    measure : function
        A function used to measure the distance between two points in the parameter space (default is Euclidean distance).
        If "paper" is given, then the measure from arXiv:XXX.XXX is used. This is ignored for categorical parameters.


    Methods
    -------
    create_initial_population(n)
        Creates an initial population of n individuals
    select_survivors(population, surviving_population_size)
        Selects the best individuals from a population based on their fitness and diversity scores
    mutation(point)
        Mutates an individual based on the specified mutation mode and rate
    run(n_generations, population_size)
        Runs the genetic algorithm for a specified number of generations, printing the average fitness at specified intervals
    """
    def __init__(self, fitness_function, gene_ranges, number_of_genes=None, crossover_method="Either Or", mutation_mode=None, mutation_rate=None, measure=None):
        # User-defined function to calculate fitness score of each individual
        self.fitness_function = fitness_function 

        # Parameter ranges of genes
        self.gene_ranges = gene_ranges

        # Function to check if parameters are a 1D list. If true, the parameters are treated as categories
        def is_one_dimensional(lst):
            return not any(isinstance(i, tuple) for i in lst)
        
        # Store results of parameter check
        self.is_discrete = is_one_dimensional(gene_ranges)

        # If parameters are categories, we print out the corresponding notice
        if self.is_discrete:
            print("Detected categorical genes.")

        # Raise error if number of parameters is not provided for categorical parameters
        if self.is_discrete and number_of_genes is None:
            raise ValueError("Your gene_ranges is a list of values, which assumes categorical genes but you have not given the number of genes in each individual with the number of parameters.")
  
        # For categorical parameters, we use their provided count. For numerical parameters, we infer the count from the parameter ranges
        self.number_of_genes = number_of_genes if number_of_genes else len(gene_ranges)
  
        # Set default mutation mode based on gene type
        default_mutation_mode = ["additive"]*self.number_of_genes if not self.is_discrete else ["categorical"]*self.number_of_genes
        self.mutation_mode = [mode.lower() for mode in mutation_mode] if mutation_mode else default_mutation_mode

        # Check if mutation methods are valid
        for mode in self.mutation_mode:
            if mode not in {'additive', 'multiplicative', 'random', 'categorical'}:
                warnings.warn(f"Invalid mutation mode '{mode}'. Available options are: 'additive', 'multiplicative', 'random', 'categorical'. Defaulting to 'additive'!")

        self.mutation_rate = mutation_rate if mutation_rate else 1.0/self.number_of_genes

        # Map string to corresponding crossover method
        crossover_methods = {
            "between": CrossoverBetween(),
            "midpoint": CrossoverMidpoint(),
            "either or": CrossoverEitherOr(), 
            "none": "none"
        }
        if crossover_method.lower() not in crossover_methods:
            warnings.warn(f"Invalid crossover method '{crossover_method}'. Available options are: {', '.join(crossover_methods.keys())}. Defaulting to 'Between'!")
        self.crossover_method = crossover_methods.get(crossover_method.lower(), CrossoverBetween())
        
        # Set the measure function
        if not measure and not self.is_discrete:
            print("No measure given, defaulting to Euclidean measure.")
        self.measure = measure if measure else None
        self.diversity = Diversity(self.measure)

        self.mutation = Mutation(self.mutation_mode, self.mutation_rate, self.gene_ranges)
    

    def create_initial_population(self, n):
        population = []
        for _ in range(n):
            if self.is_discrete:
                # Expecting a 1D list for discrete parameters
                individual_genes = [CategoricalGene(self.gene_ranges) for _ in range(self.number_of_genes)]
            else:
                # Expecting a 2D list for continuous parameters. Each item (which is a tuple) defines the (low, high) range for a gene.
                individual_genes = [NumericGene(low, high) for low, high in self.gene_ranges]

            # Create individual, which calculates its fitness        
            individual = Individual(individual_genes, self.fitness_function)
            population.append(individual)
        return population

    # def select_survivors(self, population, surviving_population_size):
    #     # List to keep selected survivors
    #     survivors = []

    #     for i in range(surviving_population_size):
    #         # Sort the population based purely on fitness for the first individual, then use fitness - diversity_score
    #         if i == 0:
    #             for individual in population:
    #                 individual.set_diversity_score(individual.fitness)

    #         population.sort(key=lambda individual: individual.diversity_score, reverse=True)
            
    #         # Get the best survivor and remove it from population
    #         best_survivor = population.pop(0)
    #         survivors.append(best_survivor)
        
    #         # Update the diversity score of remaining individuals, don't alter the fitness
    #         for individual in population:
    #             diversity_punishment = self.diversity.compute_diversity(individual, best_survivor)
    #             individual.set_diversity_score(individual.diversity_score - diversity_punishment)

    #     return survivors

    def select_survivors(self, population, surviving_population_size):
        # List to keep selected survivors
        survivors = []

        for i in range(surviving_population_size):
            # Sort the population based purely on fitness for the first individual, then use fitness - diversity_score
            if i == 0:
                for individual in population:
                    individual.diversity_score = individual.fitness

            population = sorted(population, key=lambda individual: individual.diversity_score, reverse=True)
            
            # Get the best survivor and remove it from population
            best_survivor = population.pop(0)
            survivors.append(best_survivor)
        
            # Update the diversity score of remaining individuals, don't alter the fitness
            for individual in population:
                diversity_punishment = self.diversity.compute_diversity(individual, best_survivor)
                individual.diversity_score -= diversity_punishment

        return survivors

    def run(self, n_generations, population_size):
            # Set population size for diversity calculation
            self.diversity.set_population_size(population_size)
            # Create initial population
            population = self.create_initial_population(population_size)
            if population is None:
                raise ValueError("Failed to create initial population.")

            # Determine the generations at which to print the averages
            print_generations = np.linspace(0, n_generations, 6, dtype=int)[1:]

            # Run the genetic algorithm for the specified number of generations
            for generation in range(n_generations):
                # Apply crossover and mutation to create new population
                new_population = []
                for i in range(population_size):
                    if self.crossover_method == "none":
                        # If no crossover, take the individual directly from the current population
                        child = population[i]
                    else:
                        # Select two parents randomly
                        random_indices = np.random.choice(population_size, 2, replace=False)
                        parent1 = population[random_indices[0]]
                        parent2 = population[random_indices[1]]
                        # Create child by crossover
                        child = self.crossover_method.crossover(parent1, parent2)

                    # Apply mutation
                    child = self.mutation.mutate(child)
                    new_population.append(child)

                # Combine old and new populations
                combined_population = population + new_population

                # Select the best individuals to form the next generation
                population = self.select_survivors(combined_population, population_size)

                if generation in print_generations or generation == 0:
                    average_fitness = np.mean([individual.fitness for individual in population])
                    print(f"Generation {generation}, Average Fitness: {average_fitness}")
                
            return [individual.get_gene_values() for individual in population]