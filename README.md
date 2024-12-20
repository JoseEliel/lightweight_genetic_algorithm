## Lightweight Genetic Algorithm

<img src="https://github.com/JoseEliel/lightweight_genetic_algorithm/raw/main/graphical_abstract.png" width="800"/>

- [About](#about)
- [Installation](#installation)
- [Features](#features)
- [User Guide](#user-guide)
    - [Overview](#overview)
    - [Example 1: Numerical Genes](#example-1-numerical-genes)
    - [Example 2: Categorical Genes](#example-2-categorical-genes)


### <a id="about"></a> About

This package provides an intuitive, flexible, and efficient implementation of a genetic algorithm in Python. It is designed to be easy to use while still providing a high degree of flexibility for a wide range of optimization problems. The package is developed by Eliel Camargo-Molina and Jonas Wessén. 

> For detailed documentation please visit [Read the Docs](https://lightweight-genetic-algorithm.readthedocs.io/).


### <a id="installation"></a>Installation

The `lightweight-genetic-algorithm` Python module can be installed using `pip`:

```bash
pip install lightweight-genetic-algorithm
```

The source code is available at [github.com/JoseEliel/lightweight_genetic_algorithm](https://github.com/JoseEliel/lightweight_genetic_algorithm).

### <a id="features"></a>Features

The `lightweight-genetic-algorithm` Python module contains several features that allow the user to easily set up a GA for a wide range of optimization problems. These features include:

- **Support for Numerical and Categorical Genes**: The package can handle optimization problems formulated either in terms of numerical or categorical genes.
- **Multiple Crossover Methods**: The package provides four different crossover methods: `Between`, `Midpoint`, `Either Or`, and `None`.
- **Multiple Mutation Modes**: The package includes three mutation modes for numerical genes: `additive`, `multiplicative`, `random`. For categorical genes, the algorithm assumes the `categorical` mutation mode.
- **Diversity-Enhanced Selection**: The package uses the diversity-enhanced selection algorithm, allowing the algorithm to explore widely different regions in parameter space.
- **Customizable Distance Measure**: The user can specify the distance function (used during selection) to be either `Euclidean` or `Dynamic` for numeric genes. For categorical genes, the Hamming distance is assumed. It is also possible for the user to supply their own distance function.
- **Multiprocessing**: The package supports multiprocessing for parallel fitness evaluations, which can dramatically speed up the genetic algorithm for problems where the fitness function is computationally expensive.

## <a id="user-guide"></a> User Guide

### <a id="overview"></a> Overview

The primary class in this package is `GeneticAlgorithm`. A `GeneticAlgorithm` instance is created with the following inputs:

- `fitness_function`: A function computing the fitness score of an individual. This function should receive an array of genes as its first input argument and return a single number. Additional arguments can be passed to the fitness function using the `fitness_function_args` argument.
- `gene_ranges`: A list of tuples representing the range of each numeric gene. Each tuple should contain two numbers, with the first number being the lower bound and the second the upper bound. For categorical genes, `gene_ranges` should instead be a one-dimensional list of possible categories.
- `number_of_genes` (only needed for categorical genes): The number of genes defining an individual. For numeric genes, the `number_of_genes` is inferred from the length of `gene_ranges`.
- `fitness_function_args` (optional): Additional arguments to pass to the fitness function. This should be a tuple of arguments.
- `crossover_method` (optional): The method used for crossover. Available options are `Between`, `Midpoint`, `Either Or`, and `None`. Default is `Between` for numeric genes. For categorical genes, only `Either Or` or `None` is possible.
- `mutation_mode` (optional): The mode used for mutation. Options available are `additive`, `multiplicative`, `random`, and `categorical`. Default is `additive` for numeric genes and `categorical` for categorical genes.
- `mutation_rate` (optional): The rate of mutation. The default is 1.0/`number_of_genes`. During crossover, each gene is mutated with probability `mutation_rate`.
- `measure` (optional): Specifies the distance function between two points in the gene space. This argument can be a string variable (`Euclidean`, `Dynamic`, or `Hamming`) corresponding to the three distance measures discussed in this work. The `measure` argument can also be a user-defined distance function. The default is Euclidean distance for numeric genes and Hamming distance for categorical genes.
- `r0` (optional): The characteristic distance beyond which there is no diversity penalty (default is 1/10 of the average spread of initial population). Only used for diversity enhanced selection.
- `D0` (optional): The maximum diversity penalty for identical individuals (default is 1.0). Only used for diversity enhanced selection.
- `use_multiprocessing` (optional): Whether to use multiprocessing for parallel fitness evaluations. Default is False.
- `ncpus` (optional): The number of CPUs to use for multiprocessing. Default is the number of CPUs on the system minus one. This argument is used only when `use_multiprocessing` is True.
- `selection_method` (optional): The method used for survivor selection. Available options are `Diversity Enhanced` and `Fitness Proportionate`. Default is `Diversity Enhanced`.
- `output_directory` (optional): The directory where the output files are saved. Default is None. If specified, the algorithm saves the genes of the selected survivors in each generation to a file `<date-time>_survivors.npy` as a *numpy* array in the output directory. The average fitness and best fitness at each generation are saved to `<date-time>_fitness.txt`. A log file is saved to `log.txt` containing the output of the algorithm printed to the console.

Once an instance of the `GeneticAlgorithm` class has been created, the genetic algorithm is executed using the `run` or `run_light` methods. These methods take the following arguments:

- `n_generations`: The number of generations to run the genetic algorithm for.
- `population_size`: The number of individuals in the population.
- `fitness_threshold` (optional): The fitness threshold at which the genetic algorithm should stop. If this is set, the genetic algorithm will stop when the fitness of the best individual in the population is greater than or equal to the fitness threshold. Default is None.
- `init_genes` (optional): An initial set of gene values for creating the initial population. Default is `None`.
- `verbosity` (optional): The verbosity level for printing out messages. The options are `0` (silent), `1` (normal output), and `2` (detailed output). Default is `1`.

The `run` method returns the full list of `Individual` instances across all generations, where each `Individual` object has attributes such as fitness and gene values.

The `run_light` method is similar to `run` but returns only the gene values of all individuals across all generations.

### <a id="example-1-numerical-genes"></a> Example 1: Numerical Genes

In this example, an individual represents a point in the xy-plane, and the fitness function takes the form:

$$ f(x,y) = - A \left( \sqrt{x^2 + y^2} - R \right)^2, $$

which has an extended maximum on the circle centered at the origin with radius $R$. The overall factor $A$ can be chosen to balance the diversity punishment and the fitness reward. The goal of the algorithm is to find a set of points that are evenly distributed along the entire circle.

The complete Python code for this example is shown below. The genetic algorithm is run for 20 generations with a population size of 100 using the `Between` crossover method.

```python
from lightweight_genetic_algorithm import GeneticAlgorithm

# Define fitness function
def fitness_function(individual): # individual is x,y coordinates
    distance = (individual[0]**2 + individual[1]**2)**0.5
    R = 5 # Circle radius
    A = 5 # Overall fitness scaling 
    fitness = -A*(distance - R)**2 
    return fitness

# Define the ranges of the genes
gene_ranges = [ (-10,10), (-10,10) ]

# Create a GeneticAlgorithm instance
ga = GeneticAlgorithm(fitness_function, gene_ranges, crossover_method='Between')
all_populations = ga.run_light(n_generations=20, population_size=100)

# all_populations is a (n_generations, population_size, n_genes) list.
# The final population is all_populations[-1]
```

*Code for the categorical genes example. In this example, an individual represents coordinates of a point in the xy-plane. The fitness function has an extended maximum on a circle with radius 5.0. The genetic algorithm is run for 20 generations with a population size of 100 using the `Between` crossover method.*

The resulting population after 20 generations is depicted below in Figure 1. In 20 generations, the individuals are evenly distributed along the circle.

![Figure 1: Initial and final populations for the numerical genes example.](https://github.com/JoseEliel/lightweight_genetic_algorithm/raw/main/Figure_A2.png)

*Figure 1: The initial population is randomly distributed, while the final population is evenly distributed along the circle after 20 generations.*


### <a id="example-2-categorical-genes"></a> Example 2: Categorical Genes

Next, we turn to a slightly more complex example involving categorical genes. In this example, we seek to construct an array of Lysine (K) and Glutamic acid (E) representing the amino-acid sequence of a model intrinsically disordered protein. The goal is to find a diverse set of sequences with a sequence charge decoration (SCD) parameter near a given target value.

The net charge of a sequence is the sum of the charges of the amino acids with Lysine (K) having a charge of +1 and Glutamic Acid (E) having a charge of -1. The SCD parameter is a single number that can be calculated given a sequence of charges. The SCD parameter is a measure of the "charge blockiness" (i.e., an alternating sequence `EKEKEK...EK` has SCD ≈ 0 while a di-block sequence `EEEE...EEEKKKK...KKK` gives a large, negative SCD) and correlates well with both the radius-of-gyration of isolated chains and with the upper-critical temperature for phase separation in multi-chain systems.

The complete Python code for this example is shown in below. In this example, an individual corresponds to a list of `E`'s and `K`'s representing the amino-acid sequence. This code showcases two additional important features of the `lightweight-genetic-algorithm` module: multiprocessing and the usage of additional arguments to the fitness function.

```python
from lightweight_genetic_algorithm import GeneticAlgorithm

# Calculates the sequence charge decoration (SCD) parameter
def calculate_SCD(sequence):
    aa_charges = {'K':1, 'E':-1} # Amino acid electric charges
    charge_sequence = [ aa_charges[aa] for aa in sequence ]
    
    SCD = 0
    for a in range(len(charge_sequence)-1):
        for b in range(a+1,len(charge_sequence)):
            SCD += charge_sequence[a] * charge_sequence[b] * (b-a)**0.5
    SCD /= len(charge_sequence)
    return SCD

# Define fitness function
def fitness_function(sequence, target_SCD):
    SCD = calculate_SCD(sequence)
    fitness = -(SCD - target_SCD)**2
    return fitness

def main():
    # Define the ranges of the genes. Categorical genes are recognized automatically since it is a one-dimensional list. 
    gene_ranges = ['E', 'K'] # Glutamic acid (E, charge=-1) , Lysine (K, charge=+1)

    N = 50 # sequence length
    target_SCD = -10 # Target SCD value

    # Create a GeneticAlgorithm instance
    ga = GeneticAlgorithm(fitness_function, gene_ranges, 
                        number_of_genes = N, 
                        fitness_function_args = (target_SCD,), 
                        use_multiprocessing = True)

    # Run the genetic algorithm
    all_populations = ga.run_light(n_generations=50, population_size=100)

if __name__ == '__main__':
    main()
```

*Complete Python code for the categorical genes example, showcasing the usage of multiprocessing and additional fitness function arguments. The GA is run for 50 generations with a population size of 100. The final population of sequences is contained in `all_populations[-1]` which is a list of length 100 where each entry is a list of `E`'s and `K`'s representing the amino-acid sequence.*

The net charges and SCD values for the initial and final populations are shown in Figure 2 below. Note that the SCD values are close to the target value of -10 while there is a wide range of net charges in the final population. This demonstrates the effect of the diversity-enhanced selection method.

![Figure 2: SCD values and net charges for initial and final populations of the categorical genes example.](https://github.com/JoseEliel/lightweight_genetic_algorithm/raw/main/Figure_A4.png)

*Figure 2: The initial population has a wide range of SCD values and net charges, while the final population has SCD values close to the target value of -10.0.*
