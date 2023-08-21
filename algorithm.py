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
    survival_function : function
        A function that calculates the survival score of an individual
    param_ranges : list
        A list of tuples representing the range of each parameter; if parameters are categorical, it's a simple 1D list of categories
    crossover_method : str
        The method used for crossover (default is "Either Or")
    number_of_parameters : int
        The number of parameters (default is the length of param_ranges). This must be explicitly provided if parameters are categorical.
    mutation_mode : list
        The mode used for mutation (default is "additive" for numeric parameters and "categorical" for categorical parameters)
    mutation_rate : float
        The rate of mutation (default is 1.0/number_of_parameters)
    diversity : Diversity
        An instance of the Diversity class used to calculate diversity scores
    measure : function
        A function used to measure the distance between two points in the parameter space (default is Euclidean distance).
        If "paper" is given, then the measure from arXiv:XXX.XXX is used. This is ignored for categorical parameters.


    Methods
    -------
    create_initial_population(n)
        Creates an initial population of n individuals
    survival(point)
        Calculates the survival score of an individual
    select_survivors(population, surviving_population_size)
        Selects the best individuals from a population based on their survival scores and diversity scores
    mutation(point)
        Mutates an individual based on the specified mutation mode and rate
    run(n_generations, population_size)
        Runs the genetic algorithm for a specified number of generations, printing the average fitness at specified intervals
    """
    def __init__(self, survival_function, param_ranges, number_of_parameters=None, crossover_method="Either Or", mutation_mode=None, mutation_rate=None, measure=None):
        # User-defined function to calculate survival score of each individual
        self.survival_function = survival_function 

        # Parameter ranges of genes
        self.param_ranges = param_ranges

        # Function to check if parameters are a 1D list. If true, the parameters are treated as categories
        def is_one_dimensional(lst):
            return not any(isinstance(i, tuple) for i in lst)
        
        # Store results of parameter check
        self.is_discrete = is_one_dimensional(param_ranges)

        # If parameters are categories, we print out the corresponding notice
        if self.is_discrete:
            print("Detected categorical genes.")

        # Raise error if number of parameters is not provided for categorical parameters
        if self.is_discrete and number_of_parameters is None:
            raise ValueError("Your param_ranges is a list of values, which assumes categorical genes but you have not given the number of genes in each individual with the number of parameters.")
  
        # For categorical parameters, we use their provided count. For numerical parameters, we infer the count from the parameter ranges
        self.number_of_parameters = number_of_parameters if number_of_parameters else len(param_ranges)
  
        # Set default mutation mode based on gene type
        default_mutation_mode = ["additive"]*self.number_of_parameters if not self.is_discrete else ["categorical"]*self.number_of_parameters
        self.mutation_mode = [mode.lower() for mode in mutation_mode] if mutation_mode else default_mutation_mode

        # Check if mutation methods are valid
        for mode in self.mutation_mode:
            if mode not in {'additive', 'multiplicative', 'random', 'categorical'}:
                warnings.warn(f"Invalid mutation mode '{mode}'. Available options are: 'additive', 'multiplicative', 'random', 'categorical'. Defaulting to 'additive'!")

        self.mutation_rate = mutation_rate if mutation_rate else 1.0/self.number_of_parameters

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

        self.mutation = Mutation(self.mutation_mode, self.mutation_rate, self.param_ranges)
    
    def create_initial_population(self, n):
        population = []
        for _ in range(n):
            if self.is_discrete:
                # Expecting a 1D list for discrete parameters
                individual_genes = [CategoricalGene(self.param_ranges) for _ in range(self.number_of_parameters)]
            else:
                # Expecting a 2D list for continuous parameters. Each item (which is a tuple) defines the (low, high) range for a gene.
                individual_genes = [NumericGene(low, high) for low, high in self.param_ranges]
            
            individual = Individual(individual_genes)
            population.append(individual)
        return population

    def survival(self, individual):
        # Use the user-defined survival function
        try:
            survival = self.survival_function(individual.get_gene_values())
        except:
            raise ValueError("Error in survival function evaluation. Your survival function does not seem to be compatible with your individuals.")

        return survival


    def select_survivors(self, population, survival_scores, surviving_population_size):
        # Arrange individuals, their scores and original survival scores into tuples
        scored_population = list(zip(survival_scores, survival_scores, population))
        
        # List to keep selected survivors and their original survival scores
        survivors = []
        survivor_scores = []

        for _ in range(surviving_population_size):
            # Sort the scored population
            scored_population.sort()
            
            # Get the best survivor and its original survival score and remove it from scored_population
            _, original_score, best_survivor = scored_population.pop(0)
            survivors.append(best_survivor)
            survivor_scores.append(original_score)
            
            # Update the scores of the remaining individuals using list comprehension
            scored_population = [(score + self.diversity.compute_diversity(individual, best_survivor), original_score, individual) 
                                for score, original_score, individual in scored_population]

        return survivors, survivor_scores
    
    def run(self, n_generations, population_size):
            # Set population size for diversity calculation
            self.diversity.set_population_size(population_size)
            # Create initial population
            population = self.create_initial_population(population_size)
            if population is None:
                raise ValueError("Failed to create initial population.")

            # Determine the generations at which to print the averages
            print_generations = np.linspace(0, n_generations, 6, dtype=int)[1:]

            # Calculate and store the survival scores of the initial population
            survival_scores = [self.survival(individual) for individual in population]

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

                # Calculate the survival scores of the new population
                new_survival_scores = [self.survival(individual) for individual in new_population]

                # Combine old and new populations
                combined_population = population + new_population
                combined_survival_scores = survival_scores + new_survival_scores

                # Select the best individuals to form the next generation
                population, survival_scores = self.select_survivors(combined_population, combined_survival_scores, population_size)

                if generation in print_generations or generation == 0:
                    average_fitness = np.mean(survival_scores)
                    print(f"Generation {generation}, Average Fitness: {average_fitness}")
                
            return [individual.get_gene_values() for individual in population]