import numpy as np
from .selection import SurvivorSelection, DiversityEnhancedSurvivorSelection, FitnessProportionalSurvivorSelection
from .crossover import CrossoverBetween, CrossoverMidpoint, CrossoverEitherOr
from .mutation import Mutation
from .population import Individual, NumericGene, CategoricalGene
import warnings
import os

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
        The method used for crossover. Options are "Between", "Midpoint", "Either Or", or "None" (default is "Either Or")
    number_of_genes : int
        The number of parameters (default is the length of gene_ranges). This must be explicitly provided if parameters are categorical.
    mutation_mode : list
        The mode used for mutation (default is "additive" for numeric parameters and "categorical" for categorical parameters)
    mutation_rate : float
        The rate of mutation (default is 1.0/number_of_genes). During crossover, each gene is mutated with probability mutation_rate.
    measure : function or str
        Defines the distance between two individuals in the parameter space. String options are "Hamming", "Euclidean", "Dynamic".
        Alternatively, a custom function can be provided. This function takes the genes of two individuals and returns a distance.
    use_multiprocessing : bool
        Whether to use multiprocessing for fitness evaluations (default is False). 
        This speeds up the algorithm for computationally expensive fitness functions.
    ncpus : int
        The number of processes to use for multiprocessing (default is the number of CPUs on the system minus 1)
    verbosity : int
        The verbosity level for printing out messages (default is 1). 0 = silent, 1 = normal output, 2 = detailed output.
    selection_method : str
        The method used for selecting the best individuals for the next generation. Options are "Diversity Enhanced" or "Fitness Proportionate".
    output_directory : str
        The directory to save output files (default is None). If specified, the algorithm saves the genes of the selected survivors in each generation to a file "survivors.npy" in the output directory. The average fitness and best fitness at each generation are saved to "fitness.txt". A log file is saved to "log.txt" containing the output of the algorithm printed to the console. 

        The "survivors.npy" file can be loaded using numpy.load(file, allow_pickle=True) to access the gene values of the individuals in each generation. The loaded array has shape (n_generations+1, population_size, number_of_genes) since the initial population is also saved.

    Methods
    -------
    run(n_generations, population_size, initial_population, fitness_threshold)
        Runs the genetic algorithm for a specified number of generations, printing the average fitness at specified intervals. 
        This method returns a list of lists, where each sublist contains individuals in the population at each generation. The individuals are represented as Individual objects.
    run_light(n_generations, population_size, initial_population, fitness_threshold)
        Runs the genetic algorithm for a specified number of generations, returning only the gene values of the individuals in the population at each generation.
    """
    def __init__(self, fitness_function, gene_ranges, fitness_function_args=(), number_of_genes=None, crossover_method="Either Or", mutation_mode=None, mutation_rate=None, measure=None, use_multiprocessing=False, ncpus=None, verbosity=1, selection_method="Diversity Enhanced", output_directory=None):
        # Verbosity level for printing out messages
        self.verbosity = verbosity
        self.output_directory = output_directory
        if self.output_directory:
            self.setup_output_directory()

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
            self.log("Detected categorical genes.", level = 2)
        else:
            self.log("Detected numeric genes.", level = 2)

        # Raise error if number of parameters is not provided for categorical parameters
        if self.is_discrete and number_of_genes is None:
            raise ValueError("Your gene_ranges is a list of values, which assumes categorical genes but you have not given the number of genes in each individual with the number of parameters.")
  
        # For categorical parameters, we use their provided count. For numerical parameters, we infer the count from the parameter ranges
        self.number_of_genes = number_of_genes if number_of_genes else len(gene_ranges)
  
        # Set default mutation mode based on gene type
        default_mutation_mode = ["additive"]*self.number_of_genes if not self.is_discrete else ["categorical"]*self.number_of_genes
        self.mutation_mode = [mode.lower() for mode in mutation_mode] if mutation_mode else default_mutation_mode

        self.log(f"Mutation mode: {', '.join(self.mutation_mode)}", level=2)

        # Check if mutation methods are valid
        for mode in self.mutation_mode:
            if mode not in {'additive', 'multiplicative', 'random', 'categorical'}:
                warnings.warn(f"Invalid mutation mode '{mode}'. Available options are: 'additive', 'multiplicative', 'random', 'categorical'. Defaulting to 'additive'!")

        self.mutation_rate = mutation_rate if mutation_rate else 1.0/self.number_of_genes
        self.log(f"Mutation rate: {self.mutation_rate}", level=2)

        self.mutation = Mutation(self.mutation_mode, self.mutation_rate, self.gene_ranges)

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
        
        self.log(f"Crossover method: {crossover_method}", level=2)

        ##### Set-up the diversity enhanced survivor selection ##### 
        if selection_method.lower() == "diversity enhanced":
            self.log("Using diversity enhanced selection.", level=2)
            # (1) Set the distance measure function
            if not measure:
                if self.is_discrete:
                    self.log("No measure given, defaulting to Hamming measure.", level=2)
                    self.measure = 'hamming'
                else:
                    self.log("No measure given, defaulting to Euclidean measure.", level=2)
                    self.measure = 'euclidean'   
            elif measure == "dynamic":
                self.log("Using dynamic measure.", level=2)
                if self.is_discrete:
                    raise ValueError("Dynamic measure is not compatible with categorical parameters.")
                self.measure = 'dynamic'
            else:
                self.log(f"Using user-defined input distance measure.", level = 2)
                self.measure = measure  
            # (2) Create SurvivorSelection instance
            self.survivor_selection = DiversityEnhancedSurvivorSelection(self.measure)
        elif selection_method.lower() == "Fitness Proportionate".lower():
            self.log("Using fitness proportionate selection.", level=2)
            self.survivor_selection = FitnessProportionalSurvivorSelection()
        else:
            raise ValueError("Invalid selection method. Available options are: 'Diversity Enhanced', 'Fitness Proportionate'.")

        # Setup multiprocessing if specified
        self.use_multiprocessing = use_multiprocessing
        if self.use_multiprocessing:
            self.log("Using multiprocessing.", level=2)
            import multiprocessing as mp
            self.mp = mp
            self.ncpus = ncpus if ncpus else self.mp.cpu_count()-1
            # Initialize the multiprocessing Pool
            self.pool = self.mp.Pool(self.ncpus)
        else:
            self.log("Not using multiprocessing.", level=2)
            self.pool = None
    
    def log(self, message, level=1):
        if self.verbosity >= level:
            print(message)
        if self.output_directory:
            with open(f"{self.output_directory}/log.txt", "a") as f:
                f.write(message + "\n")

    def setup_output_directory(self):
        if self.output_directory[-1] != '/':
            self.output_directory += '/'

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            self.log(f"Output directory created at {self.output_directory}", level=1)
        else:
            files_to_remove = ['log.txt', 'fitness.txt', 'survivors.npy']
            for file in files_to_remove:
                if os.path.exists(f"{self.output_directory}/{file}"):
                    os.remove(f"{self.output_directory}/{file}")
            self.log(f"Output directory {self.output_directory} already existed. Removed any previous log.txt/fitness.txt/survivors.npy files.", level=1)

    def start_pool(self):
        if self.use_multiprocessing and self.pool is None:
            self.pool = self.mp.Pool(self.ncpus)

    def stop_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def evaluate_fitness(self,genes):
        return self.fitness_function(genes,*self.fitness_function_args)

    def create_initial_population(self, n, init_gene_values=None):
        self.log(f"Creating initial population of size {n}.", level=2)

        # Create genes of the population
        if self.is_discrete:
            # Expecting a 1D list for discrete parameters
            genes = [ [CategoricalGene(self.gene_ranges) for _ in range(self.number_of_genes)] for _ in range(n) ]
        else:
            # Expecting a 2D list for continuous parameters.
            genes = [ [NumericGene(self.gene_ranges[i])  for i in range(self.number_of_genes)] for _ in range(n) ]

        # If initial gene values are specified, use them to overwrite the initial population genes
        if init_gene_values:
            for i in range(n):
                for j in range(self.number_of_genes):
                    genes[i][j].set_value(init_gene_values[j])

        # Create population using multiprocessing if specified
        if self.use_multiprocessing:
            with self.mp.Pool(self.ncpus) as pool:
                population = pool.starmap(Individual, [(g, self.fitness_function, self.fitness_function_args) for g in genes] )
            return population
        
        # Otherwise, create population sequentially
        population = [Individual(g, self.fitness_function, self.fitness_function_args) for g in genes]
        return population


    def run(self, n_generations, population_size, init_genes=None, fitness_threshold=None, verbosity=1):
            '''
            Run the genetic algorithm and return the population at each generation. 

            Parameters
            ----------
            n_generations : int
                The number of generations to run the genetic algorithm
            population_size : int
                The number of individuals in the population
            init_genes : list
                A list of gene values to initialize the population with. If None, the population is initialized randomly.
            fitness_threshold : float
                The fitness threshold at which to stop the algorithm early. If None, the algorithm runs for n_generations.
            verbosity : int
                The verbosity level for printing out messages. 0 = silent, 1 = normal output, 2 = detailed output.

            Returns
            -------
            list
                A list of lists, where each sublist contains individuals in the population at each generation. The individuals are represented as Individual objects.

            '''
            self.verbosity = verbosity
            # Start multiprocessing pool if specified
            self.start_pool()
            # Create initial population
            population = self.create_initial_population(population_size, init_genes)
            if population is None:
                raise ValueError("Failed to create initial population.")
            
            # Set the values for r0
            if isinstance(self.survivor_selection, DiversityEnhancedSurvivorSelection):                        
                # Compute the average measure between two individuals, including only non-zero distances
                initial_population_distances = [self.survivor_selection.measure(p1.get_gene_values(), p2.get_gene_values()) for p1 in population for p2 in population]
                # drop the zeros
                initial_population_distances = [d for d in initial_population_distances if d > 0]
                # get the average distance
                avg_measure = np.mean(initial_population_distances)
                # set r0
                avg_distance = np.sqrt(avg_measure)
                self.survivor_selection.r0 = avg_distance/10
                
            
            # Determine the generations at which to print the averages
            print_generations = np.linspace(0, n_generations, 6, dtype=int)[1:]
            # Run the genetic algorithm for the specified number of generations

            historical_population = [ [ individual.copy() for individual in population ] ]
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
                    offspring = self.pool.starmap(Individual, [(g, self.fitness_function, self.fitness_function_args) for g in offspring_genes] )
                else:
                    offspring = [Individual(genes, self.fitness_function, self.fitness_function_args) for genes in offspring_genes]

                # Combine parent and offspring populations (Elitism)
                combined_population = population + offspring 

                # Select the best individuals to form the next generation
                population = self.survivor_selection.select_survivors(combined_population, population_size)

                best_fitness = np.max( [individual.fitness for individual in population] )
                if generation in print_generations or generation == 0:
                    average_fitness = np.mean([individual.fitness for individual in population])
                    self.log(f"Generation {generation}, Average Fitness: {average_fitness}, Best Fitness: {best_fitness}", level=1)

                historical_population.append( [individual.copy() for individual in population ] )

                if self.output_directory:
                    with open(f"{self.output_directory}/fitness.txt", "a") as f:
                        f.write(f"{generation} {average_fitness} {best_fitness}\n")
                    # Save the genes of the historical population to the output file.
                    np.save(f"{self.output_directory}/survivors.npy", [ [individual.get_gene_values() for individual in population] for population in historical_population ])

                # Check if fitness threshold is reached
                if fitness_threshold and best_fitness >= fitness_threshold:
                    self.log(f"Fitness threshold reached at generation {generation}!", level=1)
                    break
            # Stop multiprocessing pool if specified    
            self.stop_pool()
            
            return historical_population
    
    def run_light(self, n_generations, population_size, init_genes=None, fitness_threshold=None, verbosity=1):
        ''' 
        Run the genetic algorithm and return the gene values of the individuals in the population at each generation.

        Parameters
        ----------
        n_generations : int
            The number of generations to run the genetic algorithm
        population_size : int
            The number of individuals in the population
        init_genes : list
            A list of gene values to initialize the population with. If None, the population is initialized randomly.
        fitness_threshold : float
            The fitness threshold at which to stop the algorithm early. If None, the algorithm runs for n_generations.
        verbosity : int
            The verbosity level for printing out messages. 0 = silent, 1 = normal output, 2 = detailed output.
        '''

        historical_population = self.run(n_generations, population_size, init_genes=init_genes, fitness_threshold=fitness_threshold, verbosity=verbosity)
        return [[individual.get_gene_values() for individual in population] for population in historical_population]