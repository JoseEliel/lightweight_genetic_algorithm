import numpy as np
from .diversity import Diversity
from .crossover import CrossoverBetween, CrossoverMidpoint, CrossoverEitherOr
import matplotlib.pyplot as plt
import warnings

class GeneticAlgorithm:
    """
    A class used to represent a Genetic Algorithm

    Attributes
    ----------
    survival_function : function
        A function that calculates the survival score of an individual
    param_ranges : list
        A list of tuples representing the range of each parameter
    crossover_method : str
        The method used for crossover (default is "Between")
    number_of_parameters : int
        The number of parameters (default is the length of param_ranges)
    mutation_mode : list
        The mode used for mutation (default is "additive" for all parameters)
    mutation_rate : float
        The rate of mutation (default is 1.0/number_of_parameters)
    diversity : Diversity
        An instance of the Diversity class used to calculate diversity scores

    Methods
    -------
    create_initial_population(n)
        Creates an initial population of n individuals
    survival(point)
        Calculates the survival score of an individual
    select_survivors(population, surviving_population_size)
        Selects the best individuals from a population
    mutation(point)
        Mutates an individual
    run(n_generations, population_size)
        Runs the genetic algorithm for a specified number of generations
    """
    def __init__(self, survival_function, param_ranges, crossover_method="Between", number_of_parameters=None, mutation_mode=None, mutation_rate=None, distance=None):
        self.survival_function = survival_function
        self.param_ranges = param_ranges
        self.number_of_parameters = number_of_parameters if number_of_parameters else len(param_ranges)
        self.mutation_mode = mutation_mode if mutation_mode else ["additive"]*self.number_of_parameters
                # check if mutation methods are valid
        for mode in self.mutation_mode:
            if mode not in {'additive', 'multiplicative', 'random'}:
                warnings.warn(f"Invalid mutation mode '{mode}'. Available options are: 'additive', 'multiplicative', 'random'. Defaulting to 'additive'!")

        self.mutation_rate = mutation_rate if mutation_rate else 1.0/self.number_of_parameters
        self.diversity = Diversity(self.number_of_parameters)

        # Map string to corresponding crossover method
        crossover_methods = {
            "between": CrossoverBetween(),
            "midpoint": CrossoverMidpoint(),
            "either or": CrossoverEitherOr()
        }
        if crossover_method.lower() not in crossover_methods:
            warnings.warn(f"Invalid crossover method '{crossover_method}'. Defaulting to 'Between'. Available options are: {', '.join(crossover_methods.keys())}. Defaulting to 'Between'!")
        self.crossover_method = crossover_methods.get(crossover_method.lower(), CrossoverBetween())

    def create_initial_population(self, n):
        population = []
        for _ in range(n):
            individual = [np.random.uniform(low=low, high=high) for low, high in self.param_ranges]
            population.append(individual)
        return np.array(population)

    def survival(self, point):
        # Use the user-defined survival function
        return self.survival_function(point)

    def select_survivors(self, population, surviving_population_size):
        # Compute the survival scores for all individuals at once
        survival_scores = np.apply_along_axis(self.survival, 1, population)

        # Create a 2D array to store the index and survival score of each individual
        SurvSort = np.column_stack((np.arange(len(population)), survival_scores))

        # Initialize a 2D array to store the selected survivors
        survivors = np.zeros((surviving_population_size, len(population[0])))

        # Select the k survivors
        for i in range(surviving_population_size):
            # Sort SurvSort in ascending order based on the survival scores
            SurvSort = SurvSort[np.argsort(SurvSort[:, 1])]

            # Select the individual with the highest survival score that has not been selected yet
            survivors[i] = population[int(SurvSort[i, 0])]

            # Update the survival scores of the remaining individuals
            for j in range(i + 1, len(population)):
                # Compute the diversity score between the selected individual and the j-th individual
                diversity_score = self.diversity.compute_diversity(population[int(SurvSort[j, 0])], survivors[i], surviving_population_size)
                # Add the diversity score to the survival score of the j-th individual
                SurvSort[j, 1] += diversity_score

        # Return the selected survivors
        return survivors

    def mutation(self, point):
        # Loop over parameters. A mutation occurs by adding or multiplying a random value, depending on the specified mode. 
        for i in range(self.number_of_parameters):
            # Check if a mutation should occur based on the specified mutation rate. 
            if np.random.rand() < self.mutation_rate:
                # If the mutation mode is additive, add a random number within the range of parameter i.
                if self.mutation_mode[i] == "additive":
                    point[i] += np.random.uniform(low=self.param_ranges[i][0], high=self.param_ranges[i][1])
                # If the mutation mode is multiplicative, multiply by a random number within a certain range.
                elif self.mutation_mode[i] == "multiplicative":
                    point[i] *= np.random.uniform(low=-2, high=2)
                # If the mutation mode is random, randomly choose to add or multiply a random number 
                elif self.mutation_mode[i] == "random":
                    # Randomly choose between additive and multiplicative mutation.
                    if np.random.rand() < 0.5:
                        # (50% chance) Mutate through multiplication (multiplicative mutation)
                        point[i] *= np.random.uniform(low=-2, high=2)
                    else:
                        # (50% chance) Mutate through addition (additive mutation)
                        point[i] += np.random.uniform(low=self.param_ranges[i][0], high=self.param_ranges[i][1])
        return point
    
    def run(self, n_generations, population_size):
        # Create initial population
        population = self.create_initial_population(population_size)

        # Determine the generations at which to print the averages
        print_generations = np.linspace(0, n_generations, 6, dtype=int)[1:]

        # Run the genetic algorithm for the specified number of generations
        for generation in range(n_generations):
            # Apply crossover and mutation to create new population
            new_population = []
            for _ in range(population_size):
                # Select two parents randomly
                num_individuals = population.shape[0]
                random_indices = np.random.choice(num_individuals, 2, replace=False)
                parent1 = population[random_indices[0], :]
                parent2 = population[random_indices[1], :]
                parents = np.array([parent1, parent2])
                # Create child by crossover
                child = self.crossover_method.crossover(parents[0], parents[1])
                # Apply mutation
                child = self.mutation(child)
                new_population.append(child)

            # Combine old and new populations
            combined_population = np.concatenate((population, new_population))

            # Select the best individuals to form the next generation
            population = self.select_survivors(combined_population, population_size)

            if generation in print_generations or generation == 0:
                average_fitness = np.mean([self.survival(individual) for individual in population])
                print(f"Generation {generation}, Average Fitness: {average_fitness}")
             

        return population