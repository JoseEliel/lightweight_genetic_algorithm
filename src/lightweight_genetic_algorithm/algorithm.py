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
    fitness_function_args: tuple
        Wildcard arguments to pass to the fitness_function
    gene_ranges : list
        A list of tuples representing the range of each parameter; if parameters are categorical, it's a simple 1D list of categories
    crossover_method : str
        The method used for crossover (default is "Either Or")
    number_of_genes : int
        The number of parameters (default is the length of gene_ranges). This must be explicitly provided if parameters are categorical.
    mutation_mode : list
        The mode used for mutation (default is "additive" for numeric parameters and "categorical" for categorical parameters)
    mutation_rate : float
        The rate of mutation (default is 1.0/number_of_genes). During cross-over, each gene is mutated with probability mutation_rate.
    diversity : Diversity
        An instance of the Diversity class used to calculate diversity scores
    measure : function
        A function used to measure the distance between two points in the parameter space (default is Euclidean distance).
        If "dynamic" is given, then the dynamic measure from arXiv:XXX.XXX is used. This is ignored for categorical parameters.
    use_multiprocessing : bool
        Whether to use multiprocessing for fitness evaluations (default is False). 
        This speeds up the algorithm for computationally expensive fitness functions.
    ncpus : int
        The number of processes to use for multiprocessing (default is the number of CPUs on the system minus 1)


    Methods
    -------
    create_initial_population(n)
        Creates an initial population of n individuals
    select_survivors(population, surviving_population_size)
        Selects the best individuals from a population based on their fitness and diversity scores
    mutation(point)
        Mutates an individual based on the specified mutation mode and rate
    run(n_generations, population_size, initial_population, fitness_threshold)
        Runs the genetic algorithm for a specified number of generations, printing the average fitness at specified intervals
    """
    def __init__(self, fitness_function, gene_ranges, fitness_function_args=(), number_of_genes=None, crossover_method="Either Or", mutation_mode=None, mutation_rate=None, measure=None, use_multiprocessing=False, ncpus=None):
        # User-defined function to calculate fitness score of each individual
        self.fitness_function = fitness_function 
        self.fitness_function_args = fitness_function_args

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
        else:
            print("Detected numeric genes.")

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
        
        ##### Set-up the diversity enhanced survivor selection ##### <--- Should we define a survivor selection class instead? It would make it more modular and easier to use different selection methods.

        # Set the distance measure function
        if not measure:
            if self.is_discrete:
                print("No measure given, defaulting to Hamming measure.")
                self.measure = lambda x,y: np.sum(x != y) / len(x) 
            else:
                print("No measure given, defaulting to Euclidean measure.")
                self.measure = lambda x,y: np.sum((x - y)**2)        
        elif measure == "dynamic":
            print("Using dynamic measure.")
            if self.is_discrete:
                raise ValueError("Dynamic measure is not compatible with categorical parameters.")
            self.measure = lambda x,y: np.sum((x - y)**2 / (np.abs(x) + np.abs(y) + 1e-10)**2)
        else:
            self.measure = measure
        
        self.diversity = Diversity(self.measure)
        self.mutation = Mutation(self.mutation_mode, self.mutation_rate, self.gene_ranges)

        # Setup multiprocessing if specified
        self.use_multiprocessing = use_multiprocessing
        if self.use_multiprocessing or ncpus:
            import multiprocessing as mp
            self.mp = mp
            self.ncpus = ncpus if ncpus else self.mp.cpu_count()-1

    def evaluate_fitness(self,genes):
        return self.fitness_function(genes,*self.fitness_function_args)

    def create_initial_population(self, n):
        # Create genes of the population
        if self.is_discrete:
            # Expecting a 1D list for discrete parameters
            genes = [ [CategoricalGene(self.gene_ranges) for _ in range(self.number_of_genes)] for _ in range(n) ]
        else:
            # Expecting a 2D list for continuous parameters.
            genes = [ [NumericGene(self.gene_ranges[i])  for i in range(self.number_of_genes)] for _ in range(n) ]

        # Create population using multiprocessing if specified
        if self.use_multiprocessing:
            with self.mp.Pool(self.ncpus) as pool:
                population = pool.starmap(Individual, [(g, self.fitness_function, self.fitness_function_args) for g in genes] )
            return population
        
        # Otherwise, create population sequentially
        population = [Individual(g, self.fitness_function, self.fitness_function_args) for g in genes]
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
        '''
        Select survivors using the diversity enhancing selection method. 
        '''

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
                diversity_punishment = self.diversity.compute_diversity(individual, survivors[-1])
                diversity_scores[i] -= diversity_punishment

        return survivors

    def run(self, n_generations, population_size, fitness_threshold=None):
            '''
            Run the genetic algorithm for a specified number of generations (n_generations), printing the average and top fitness at specified intervals. The number of individuals in the population is set by population_size.
            A fitness threshold can be specified to stop the algorithm early if the fitness of the fittest individual exceeds the threshold.
            '''

            # Create initial population
            population = self.create_initial_population(population_size)
            if population is None:
                raise ValueError("Failed to create initial population.")
            
            # Set population size for diversity calculation
            self.diversity.set_population_size( len(population) )

            # Determine the generations at which to print the averages
            print_generations = np.linspace(0, n_generations, 6, dtype=int)[1:]

            # Run the genetic algorithm for the specified number of generations
            for generation in range(n_generations):                
                
                # Create genes of the offspring
                if self.crossover_method == "none":
                        # If no crossover, take the individual directly from the current population and apply mutation. 
                        # Set force_mutate to True to ensure that at least one gene is mutated.
                        offspring_genes = [self.mutation.mutate_genes(individual.get_genes(),force_mutate=True) for individual in population]
                else:
                    # Select two parents randomly
                    random_indices = np.random.choice(population_size, 2*population_size, replace=True)
                    parents = np.array(population)[random_indices].reshape(population_size, 2)
                    
                    # Create genes of the offspring by crossover
                    offspring_genes = [self.crossover_method.crossover(parent1.get_genes(), parent2.get_genes()) for parent1, parent2 in parents]

                    # Apply mutation
                    offspring_genes = [self.mutation.mutate_genes(genes) for genes in offspring_genes]

                # Create offspring Individual objects using multiprocessing if specified
                if self.use_multiprocessing:
                    with self.mp.Pool(self.ncpus) as pool:
                        # Create offspring Individual objects
                        offspring = pool.starmap(Individual, [(g, self.fitness_function, self.fitness_function_args) for g in offspring_genes] )
                else:
                    offspring = [Individual(genes, self.fitness_function, self.fitness_function_args) for genes in offspring_genes]

                # Combine parent and offspring populations (Elitism)
                combined_population = population + offspring

                # Select the best individuals to form the next generation
                population = self.select_survivors(combined_population, population_size)

                if generation in print_generations or generation == 0:
                    average_fitness = np.mean([individual.fitness for individual in population])
                    best_fitness = np.max( [individual.fitness for individual in population] )
                    print(f"Generation {generation}, Average Fitness: {average_fitness}, Best Fitness: {best_fitness}")
                
                # Check if fitness threshold is reached
                if fitness_threshold and best_fitness >= fitness_threshold:
                    print(f"Fitness threshold reached at generation {generation}!")
                    break
             
            return [individual.get_gene_values() for individual in population]