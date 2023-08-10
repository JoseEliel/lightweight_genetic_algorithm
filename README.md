# Lightweight Genetic Algorithm

This package provides an intuitive, flexible, and efficient implementation of a genetic algorithm in Python. It is designed to be easy to use while still providing a high degree of flexibility for a wide range of optimization problems.

## Features

- Supports multiple crossover methods: "Between", "Midpoint", and "Either Or".
- Allows for additive mutation.
- Includes a diversity mechanism to maintain a diverse population.
- The survival function can be customized to suit your specific optimization problem.

## Installation

You can install the package via pip:

```bash
pip install lightweight_genetic_algorithm
```

## Usage

Here is a basic example of how to use the package:

```python
from simple_genetic_algorithm import GeneticAlgorithm

# Define your survival function
def survival_function(individual):
    # This is just an example. Replace with your actual survival function.
    return sum(individual)

# Define the ranges for your parameters
param_ranges = [(0, 1), (0, 1), (0, 1)]

# Create the genetic algorithm
ga = GeneticAlgorithm(survival_function, param_ranges)

# Run the genetic algorithm
population = ga.run(n_generations=100, population_size=50)

# The final population is returned. You can extract the best individual, etc.
```

### Creating the GeneticAlgorithm class

The `GeneticAlgorithm` class is the main class of the package. It is initialized with the following parameters:

- `survival_function`: A function that takes an individual (a list of parameter values) and returns a fitness score. The higher the score, the better the individual. This is a required parameter.
- `param_ranges`: A list of tuples, where each tuple represents the lower and upper bounds for a parameter. This is a required parameter.
- `crossover_method` (optional): A string that specifies the crossover method to use. Can be "Between", "Midpoint", or "Either Or". Default is "Between".
- `number_of_parameters` (optional): The number of parameters in an individual. If not provided, it is inferred from the length of `param_ranges`.
- `additive_mutation` (optional): A list of booleans that specify whether additive mutation should be applied to each parameter. If not provided, no additive mutation is applied.
- `mutation_rate` (optional): The probability of mutation for each parameter. If not provided, it is set to 1 divided by the number of parameters.

### Running the Genetic Algorithm

The genetic algorithm is run using the `run` method of the `GeneticAlgorithm` class. It takes the following parameters:

- `n_generations`: The number of generations to run the algorithm for. This is a required parameter.
- `population_size`: The size of the population in each generation. This is a required parameter.

The `run` method returns the final population after running the algorithm for the specified number of generations.

## License

This project is licensed under the MIT License.