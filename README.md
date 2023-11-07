## Lightweight Genetic Algorithm

### About

This package provides an intuitive, flexible, and efficient implementation of a genetic algorithm in Python. It is designed to be easy to use while still providing a high degree of flexibility for a wide range of optimization problems. The package is developed by Eliel Camargo-Molina and Jonas Wessén.

The genetic algorithm implemented in this package includes features such as multiple crossover methods, mutation modes, support for both numerical and categorical genes, and a unique diversity calculation that makes it effective even for small populations and few generations.

In the context of this genetic algorithm, a "gene" can be understood as a parameter or variable that we want to optimize. It can be numeric when we explore the parameter space of a model or theory or categorical when we express discrete, non-numeric options, for example, the amino-acid sequence of in a protein.

### Installation

You can install the package using pip:

```bash
pip install lightweight-genetic-algorithm
```

### Usage

The primary class in this package is `GeneticAlgorithm`. A `GeneticAlgorithm` instance is created with the following inputs:

- `fitness_function`: A function computing the fitness score of an individual. This function should receive an array of genes as its first input argument and return a single number. Additional arguments can be passed to the fitness function using the `fitness_function_args` argument, described below.

- `gene_ranges`: A list of tuples representing the range of each numeric gene. Each tuple should contain two numbers, with the first number being the lower bound and the second the upper bound. For categorical genes, it should be a one-dimensional list of possible categories.

- `number_of_genes` (only needed for categorical genes): The number of genes defining an individual. For numeric genes, the `number_of_genes` is inferred from the length of `gene_ranges`.

- `fitness_function_args` (optional): Additional arguments to pass to the fitness function. This should be a tuple of arguments.

- `crossover_method` (optional): The method used for crossover. Available options are "Between", "Midpoint", "Either Or", and "None". For categorical genes, "Either Or" must be used. Default is "Between".

- `mutation_mode` (optional): The mode used for mutation. Options available are "additive", "multiplicative", "random", and "categorical". Default is "additive" for numeric genes and "categorical" for categorical genes.

- `mutation_rate` (optional): The rate of mutation. The default is 1.0/number_of_genes. During crossover, each gene is mutated with probability mutation_rate.

- `measure` (optional): A function to measure the distance between two points in the gene space. The default is Euclidean distance for numeric genes and Hamming distance for categorical genes.

- `use_multiprocessing` (optional): Whether to use multiprocessing for parallelized fitness evaluations. Default is False.

- `ncpus` (optional): The number of CPUs to use for multiprocessing. Default is the number of CPUs on the system minus one. This argument is only used if `use_multiprocessing` is True.

Once an instance of the `GeneticAlgorithm` class has been created, the genetic algorithm is executed using the `run` method. The `run` method takes the following inputs:

- `n_generations`: The number of generations to run the genetic algorithm for.

- `population_size`: The number of individuals in the population.

- `fitness_threshold` (optional): The fitness threshold at which the genetic algorithm should stop. If this is set, the genetic algorithm will stop when the fitness of the best individual in the population is greater than or equal to the fitness threshold. Default is None.

#### Example 1: Numerical genes

Here's a toy example of how you use the package to run a genetic algorithm. In this example, we seek to find a population of points that are close to a circle with a radius of 5.0. The fitness function is defined as the negative squared distance from the circle. The genes are the x and y coordinates of the points.

```python
# Define the center and radius of a circle
center = np.array([0.0, 0.0])
radius = 5.0

# Define your fitness function
def fitness_function(individual):
    distance = np.linalg.norm(individual - center)
    fitness = -abs(distance - radius)**2
    return fitness

# Define the range of your genes
gene_ranges = [(-10, 10), (-10, 10)]

# Create a GeneticAlgorithm instance
ga = GeneticAlgorithm(fitness_function, 
                      gene_ranges, 
                      crossover_method="Between",
                      number_of_genes=2, 
                      mutation_mode=["Additive", "Multiplicative"], 
                      mutation_rate=0.1)

# Run the genetic algorithm
population = ga.run(n_generations=20, population_size=100)

# Plot the final population
plt.figure(figsize=(6, 6))
plt.scatter(population[:, 0], population[:, 1], color='blue')
circle1 = plt.Circle(center1, radius1, fill=False, color='red')
plt.gca().add_artist(circle1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
```

![Image showing the resulting populations](example.png)

In this example, we're using the genetic algorithm to approximate a circular shape based on a defined radial fitness function. The genes for the `GeneticAlgorithm` class can be adjusted to fit the needs of your specific problem, including both numerical and categorical genes.

#### Example 2: Categorical genes

Here's an example showing the intended usage for categorical genes. In this example, we seek to construct an array of Lysine (K) Glutamic acid (E) representing the amino-acid sequence of a model intrinsically disordered protein. The goal is to find a diverse set of sequences with a sequence charge decoration (SCD) parameter near a given target value (target_SCD). The net charge of a sequence is the sum of the charges of the amino acids with Lysine (K) having a charge of +1 and Glutamic Acid (E) having a charge of -1. The SCD parameter is defined in Sawle & Gosh J. Chem. Phys. 143, 085101 (2015), and is a single number that can be calculated given a sequence of charges. The SCD parameter is a measure of the "charge blockiness" (i.e., an alternating sequence 'EKEKEK...EK' gives SCD~0 while a di-block sequence 'EEEE...EEEKKKK...KKK' gives a large, negative SCD) and correlates well with both the radius-of-gyration of isolated chains and with the upper-critical temperature for phase separation in multi-chain systems. 

```python
from lightweight_genetic_algorithm import GeneticAlgorithm

# Convert a amino-acid sequence to a charge sequence
def fasta_to_charge(fasta_sequence):
    charge_sequence = []
    for aa in fasta_sequence:
        if aa=='K' or aa=='R': # Lysine (K) or Arginine (R)
            charge_sequence.append(1)
        elif aa=='E' or aa=='D': # Glutamic acid (E) or Aspartic acid (D)
            charge_sequence.append(-1)
        elif aa=='H': # Histidine (H)
            charge_sequence.append(0.5)
        else:
            charge_sequence.append(0)
    return charge_sequence
    
# Calculates the sequence charge decoration (SCD) parameter
def calculate_SCD(charge_sequence):
    SCD = 0
    for a in range(len(seq)-1):
        for b in range(a+1,len(seq)):
            SCD += charge_sequence[a] * charge_sequence[b] * np.sqrt(np.abs(a-b))
    SCD *= 2/len(charge_sequence)
    return SCD

# Define fitness function
def fitness_function(fasta_sequence, target_SCD):
    charge_sequence = fasta_to_charge(fasta_sequence)
    SCD = calculate_SCD(charge_sequence)
    f = -(SCD-target_SCD)**2 
    return f

# Because it is a one-dimensional list, categorical genes are automatically recognized.  
# gene_ranges is then the list of categories, which in this example is the list of available amino acids.
gene_ranges = [ 'E', 'K' ] # Glutamic acid ('E'), Lysine ('K'),

N = 50 # sequence length
target_SCD = -15

# Create a GeneticAlgorithm instance
ga = lga.GeneticAlgorithm(fitness_function, gene_ranges, 
                          number_of_genes = N, 
                          fitness_function_args = (target_SCD,)
)

# Run the genetic algorithm
 population = ga.run(n_generations=300, population_size=100)

```
The genetic algorithm produces a population of sequences with SCD values and net charges as depicted below.

<img src="example_readme_categorical.png" width="800"/>

We can print the 10 sequences with SCD values closest to the target value using the following code:

```python
# Keep the 10 best sequences
population = population[:10]

# Calculate their SCD values and net charges
charge_sequences = [ fasta_to_charge(s) for s in population ]
all_SCD = np.array([ calculate_SCD(s) for s in charge_sequences])
all_net_charges = np.array([ np.sum(s) for s in charge_sequences])

# Print the n_print best sequences
for i in range(n_print):
    print(f"Sequence {i}:",''.join(population[i]),'SCD:',np.round(all_SCD[i],decimals=2), 'Net charge:', all_net_charges[i])
```

The output is shown below. The sequences are all different from each other but have SCD values near the target value. This is a consequence of the diversity enhancement of the genetic algorithm.

```bash
Sequence 0: KKKKKKEKKKKEKKKEEKEEEEKEKEKEEKEKEKKEEEEEKEEEEEEEEK SCD: -15.01 Net charge: -4
Sequence 1: EEEEEEKEEEKEKEEEKEEKEKKEEKEKKEKKKEKKKKEEEEKKKKKKKK SCD: -15.02 Net charge: 0
Sequence 2: KKKEKKKEKEKKKKEKKEKEKEKEKKKKEEEKEEEEEEEEKKEKEEEEEE SCD: -14.98 Net charge: -2
Sequence 3: KKKKEKEKKKKKKEKEKEKKKEEEEEKEEKEKKKKKEEEEEEKEEEEEEE SCD: -14.96 Net charge: -2
Sequence 4: KEEEEEEEEEEEEKEKKKKKEKEEKKKKEKEEEKKEEKKKKKKKEKKKKK SCD: -15.06 Net charge: 4
Sequence 5: KKKKKKKKKKKKKKKKKKKKKEKEEKEKKEKKKKEKEEKKEEEEEEEEEE SCD: -15.06 Net charge: 14
Sequence 6: KKEKKKKKKEEKKKKKKEEEEKKEKKEKKKEKEKEEKEKKEEEEEEEEEE SCD: -14.9 Net charge: 2
Sequence 7: KKKKEEEKKKKKKKKEEEEKKKKEKKKEEEEKEEEEEEEEEEKEKEEEEE SCD: -15.12 Net charge: -6
Sequence 8: EKKEKEEKEEEEEEEKEEKEKEEEEEEEKEKEKKKKKKEKKKEKKKKKEK SCD: -14.83 Net charge: -2
Sequence 9: KKKKEKEKKEKEKKKKEKKKEEKEEEEKKKKEEEEEEEKEEEEEEEEEEE SCD: -15.15 Net charge: -8
```

The above example also shows the use of the `fitness_function_args` argument, which allows you to pass additional arguments to the fitness function. In this case, we pass the target SCD value to the fitness function.

### Crossover Methods

The package provides four different crossover methods:

- **Between**: In this method, the child's genes are chosen to be a random value between the corresponding genes of the two parents. This method is suitable for numeric genes.

- **Midpoint**: In this method, the child's genes are chosen to be the average of the corresponding genes of the two parents. This method is also suitable for numeric genes.

- **Either Or**: In this method, the child's genes are chosen to be either the corresponding gene of the first parent or the second parent. This method is suitable for both numeric and categorical genes. However, for categorical genes, this is the only method that should be used.

- **None**: No crossover is performed. This method can be used when you want to run the genetic algorithm without any crossover.

### Diversity Enhanced Survivor Selection

The package includes a unique selection procedure that promotes diversity among the surviving individuals in each generation (i.e., the individuals that are selected to constitute the next-generation population). This leads to an efficient non-local exploration of the gene space, which is particularly useful for problems with many local optima. The diversity enhancement is achieved by calculating a diversity score for each individual in the population. The selection procedure works as follows:

1. The individual in the population with the highest (most optimal) fitness is selected as a survivor and removed from the population.

2. A "diversity punishment" is subtracted from the fitness of each remaining individuals. The diversity punishment is based on similarity with the previously selected survivor, with similar individuals receiving a higher punishment. 

3. Steps 1 and 2 are iterated until a new population of the desired size has been selected.

The diversity punishment is calculated using a "measure" function that defines a distance between two points in the gene space. The default measure function is the Euclidean distance for numeric genes and the Hamming distance for categorical genes. Given the distance between two individuals, the diversity punishment is obtained using an exponential function that leaves the fitness essentially unchanged for individuals further apart than certain distance. 

The package also allows you to supply your own function to measure the distance between two points in the gene space. This allows you to customize the diversity calculation to better match your problem. 

### Features

- **Multiple Crossover Methods**: The package provides four different crossover methods: "Between", "Midpoint", "Either Or", and "None". For categorical genes, "Either Or" must be used. For numeric genes, you can select the one that best matches your problem.

- **Multiple Mutation Modes**: The package includes four mutation modes: "additive", "multiplicative", "random", and "categorical". This flexibility allows you to choose the mutation mode that best suits your problem, whether it uses numeric or categorical genes.

- **Diversity Enhancement**: The package contains a unique diversity calculation which makes the algorithm effective even for small populations and a few generations. This calculation is based on the Euclidean distance between individuals, and it is designed to promote diversity within the population. The diversity score is a measure of how different two individuals are from each other, based on their gene values. The class also includes a method to adjust the diversity calculation based on the size of the total population.

- **Support for Numerical and Categorical Genes**: The GeneticAlgorithm class can handle both numeric and categorical genes, which enables users to solve a wide range of optimization problems.

- **Customizable Measure Function**: The package allows you to supply your own function to measure the distance between two points in the gene space. This flexibility lets you customize the diversity calculation to better match your problem. Note that this feature does not apply when working with categorical genes.

- **Multiprocessing**: The package supports multiprocessing for parallelized fitness evaluations. This feature can dramatically speed up the genetic algorithm for problems where the fitness function is computationally expensive. 
