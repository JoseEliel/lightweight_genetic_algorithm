# README.md

## Lightweight Genetic Algorithm

This package provides an intuitive, flexible, and efficient implementation of a genetic algorithm in Python. It is designed to be easy to use while still providing a high degree of flexibility for a wide range of optimization problems. The package is developed by Eliel Camargo-Molina and Jonas Wessén.

The genetic algorithm implemented in this package includes features such as multiple crossover methods, mutation modes, and a unique diversity calculation that makes it effective even for small populations and few generations.

### Installation

You can install the package using pip:

```bash
pip install lightweight-genetic-algorithm
```

### Usage

The main class in the package is `GeneticAlgorithm`. Here is an example of how to use it:

```python
from lightweight_genetic_algorithm import GeneticAlgorithm

# Define your survival function
def survival_function(individual):
    return sum(individual)

# Define the range of your parameters
param_ranges = [(0, 1), (0, 1), (0, 1)]

# Create a GeneticAlgorithm instance
ga = GeneticAlgorithm(survival_function, param_ranges)

# Run the genetic algorithm
population = ga.run(n_generations=100, population_size=50)
```

### Inputs

The `GeneticAlgorithm` class takes the following inputs:

- `survival_function`: A function that calculates the survival score of an individual. This function should take a list of parameters as input and return a single number.

- `param_ranges`: A list of tuples representing the range of each parameter. Each tuple should contain two numbers, with the first number being the lower bound and the second number being the upper bound.

- `crossover_method` (optional): The method used for crossover. Available options are "Between", "Midpoint", "Either Or", and "None". Default is "Between".
    - *Between*: In this method, the child's genes are chosen to be a random value between the corresponding genes of the two parents. 
    - *Midpoint*: In this method, the child's genes are chosen to be the average of the corresponding genes of the two parents. 
    - *Either Or*: In this method, the child's genes are chosen to be either the corresponding gene of the first parent or the second parent.
    - *None*: No crossover is used, new individuals are created by mutating the previous generation.

- `number_of_parameters` (optional): The number of parameters. Default is the length of `param_ranges`.

- `mutation_mode` (optional): The mode used for mutation. Available options are "additive", "multiplicative", and "random". Default is "additive" for all parameters.
  * *Additive*: In this mode, a random value within the range of the parameter is added to the gene. 
  * *Multiplicative*: In this mode, the gene is multiplied by a random value between -2 and 2. 
  * *Random*: In this mode, the algorithm randomly chooses between additive and multiplicative mutation for each gene.

- `mutation_rate` (optional): The rate of mutation. Default is 1.0/number_of_parameters.

- `measure` (optional): A function used to measure the distance between two points in the parameter space. Default is Euclidean distance.

### Features

- **Multiple Crossover Methods**: The package provides four different crossover methods: "Between", "Midpoint", "Either Or", and "None". This allows you to choose the method that best suits your problem.

- **Multiple Mutation Modes**: The package provides three different mutation modes: "additive", "multiplicative", and "random". This allows you to choose the mode that best suits your problem.

- **Diversity Calculation**: The package includes a unique diversity calculation that makes the algorithm effective even for small populations and few generations. This calculation is based on the Euclidean distance between individuals and is designed to promote diversity in the population.

- **Customizable Measure Function**: You can provide your own function to measure the distance between two points in the parameter space. This allows you to customize the diversity calculation to better suit your problem.