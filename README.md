# README.md

## Lightweight Genetic Algorithm

This package provides an intuitive, flexible, and efficient implementation of a genetic algorithm in Python. It is designed to be easy to use while still providing a high degree of flexibility for a wide range of optimization problems. The package is developed by Eliel Camargo-Molina and Jonas Wessén.

The genetic algorithm implemented in this package includes features such as multiple crossover methods, mutation modes, support for both numerical and categorical parameters, and a unique diversity calculation that makes it effective even for small populations and few generations.

### Installation

You can install the package using pip:

```bash
pip install lightweight-genetic-algorithm
```

### Usage

The main class in the package is `GeneticAlgorithm`. Here is an example of how to use it:

```python
# Define the center and radius of a circle
center = np.array([0.0, 0.0])
radius = 5.0

# Define your survival function
def survival_function(point):
    distance = np.linalg.norm(point - center)
    fitness = abs(distance - radius)**2
    return fitness

# Define the range of your parameters
param_ranges = [(-10, 10), (-10, 10)]

# Create a GeneticAlgorithm instance
ga = GeneticAlgorithm(survival_function, 
                      param_ranges, 
                      crossover_method="Between",
                      number_of_parameters=2, 
                      mutation_mode=["Additive", "Multiplicative"], 
                      mutation_rate=0.1)

# Run the genetic algorithm
population = ga.run(n_generations=75, population_size=100)

# Plot the final population
plt.figure(figsize=(6, 6))
plt.scatter(population[:, 0], population[:, 1], color='blue')
circle1 = plt.Circle(center1, radius1, fill=False, color='red')
plt.gca().add_artist(circle1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
```

**Initial population vs 75th generation**

![Image showing the resulting populations](example.png)

In this example, we use the genetic algorithm to approximate a circular shape based on a defined radial fitness function. The parameters for the `GeneticAlgorithm` class can be adjusted to fit the needs of your specific problem, including support for both numerical and categorical parameters.

### Inputs

The `GeneticAlgorithm` class takes the following inputs:

- `survival_function`: A function that calculates the survival score of an individual. This function should take a list of parameters as input and return a single number.

- `param_ranges`: A list of tuples representing the range of each numeric parameter. Each tuple should contain two numbers, with the first number being the lower bound and the second number being the upper bound. For categorical parameters, this should be a one-dimensional list of possible categories.

- `crossover_method` (optional): The method used for crossover. Available options are "Between", "Midpoint", "Either Or", and "None". Default is "Between".

- `number_of_parameters` (optional): The number of parameters. This is required when using categorical parameters. Default is the length of `param_ranges`.

- `mutation_mode` (optional): The mode used for mutation. Available options are "additive", "multiplicative", "random", and "categorical". Default is "additive" for numeric parameters and "categorical" for categorical parameters.

- `mutation_rate` (optional): The rate of mutation. Default is 1.0/number_of_parameters.

- `measure` (optional): A function used to measure the distance between two points in the parameter space. Default is Euclidean distance. If no measure is given the default is Euclidean measure. This is ignored for categorical parameters.

### Features

- **Multiple Crossover Methods**: The package provides four different crossover methods: "Between", "Midpoint", "Either Or", and "None". This allows you to choose the method that best suits your problem.

- **Multiple Mutation Modes**: The package provides four different mutation modes: "additive", "multiplicative", "random", and "categorical". This allows you to choose the mode that best suits your problem, whether using numeric or categorical parameters.

- **Diversity Calculation**: The package includes a unique diversity calculation that makes the algorithm effective even for small populations and few generations. This calculation is based on the Euclidean distance between individuals and is designed to promote diversity in the population.

- **Support for Numerical and Categorical Parameters**: The GeneticAlgorithm class can handle both numeric and categorical parameters, allowing it to be used for a wider range of optimization problems.

- **Customizable Measure Function**: You can provide your own function to measure the distance between two points in the parameter space. This allows you to customize the diversity calculation to better suit your problem. Note that this is not used when handling categorical parameters.