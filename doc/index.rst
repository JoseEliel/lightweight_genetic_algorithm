Lightweight Genetic Algorithm
=============================


.. image:: ../graphical_abstract.png
   :width: 800px

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   lightweight_genetic_algorithm
   lightweight_genetic_algorithm/algorithm
   lightweight_genetic_algorithm/crossover
   lightweight_genetic_algorithm/mutation
   lightweight_genetic_algorithm/population
   lightweight_genetic_algorithm/selection

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About
-----

The **lightweight-genetic-algorithm** package provides an intuitive, flexible, and efficient implementation of a genetic algorithm in Python. It is designed to be easy to use while providing a high degree of flexibility for a wide range of optimization problems.

*Developed by Eliel Camargo-Molina and Jonas Wessén.*

Installation
------------

You can install the package using `pip`:

.. code-block:: bash

   pip install lightweight-genetic-algorithm

The source code is available on `GitHub <https://github.com/JoseEliel/lightweight_genetic_algorithm>`_.

Features
--------

The **lightweight-genetic-algorithm** package includes several features that allow users to easily set up a genetic algorithm for a wide range of optimization problems:

- **Support for Numerical and Categorical Genes**: Handles optimization problems formulated with either numerical or categorical genes.
- **Multiple Crossover Methods**: Provides four different crossover methods: ``Between``, ``Midpoint``, ``Either Or``, and ``None``.
- **Multiple Mutation Modes**: Includes three mutation modes for numerical genes: ``additive``, ``multiplicative``, and ``random``. For categorical genes, a ``categorical`` mutation mode is used.
- **Diversity-Enhanced Selection**: Utilizes a diversity-enhanced selection algorithm, allowing exploration of widely different regions in parameter space.
- **Customizable Distance Measure**: Users can specify the distance function (used during selection) as either ``Euclidean`` or ``Dynamic`` for numerical genes. For categorical genes, the Hamming distance is used. Custom distance functions can also be provided.
- **Multiprocessing**: Supports multiprocessing for parallel fitness evaluations, which can significantly speed up the genetic algorithm for computationally expensive fitness functions.

User Guide
----------

Overview
~~~~~~~~

The primary class is ``GeneticAlgorithm``. An instance is created with the following inputs:

- **fitness_function**: A function that computes the fitness score of an individual. It should accept an array of genes as its first argument and return a single number. Additional arguments can be passed via ``fitness_function_args``.
- **gene_ranges**: For numerical genes, a list of tuples representing the range of each gene. For categorical genes, a one-dimensional list of possible categories.
- **number_of_genes** *(for categorical genes)*: The number of genes defining an individual. For numerical genes, this is inferred from ``gene_ranges``.

Optional arguments include:

- **fitness_function_args**: Additional arguments for the fitness function.
- **crossover_method**: Method used for crossover (default is ``Between`` for numerical genes).
- **mutation_mode**: Mode used for mutation (default is ``additive`` for numerical genes, ``categorical`` for categorical genes).
- **mutation_rate**: Rate of mutation (default is ``1.0/number_of_genes``).
- **measure**: Specifies the distance function between points in gene space (default is Euclidean distance for numerical genes, Hamming distance for categorical genes).
- **use_multiprocessing**: Whether to use multiprocessing for parallel fitness evaluations (default is ``False``).
- **ncpus**: Number of CPUs to use for multiprocessing.
- **selection_method**: Method used for survivor selection (default is ``Diversity Enhanced``).
- **output_directory**: Directory where output files are saved.

The genetic algorithm is executed using the ``run`` or ``run_light`` methods, which require:

- **n_generations**: Number of generations to run.
- **population_size**: Number of individuals in the population.

Optional arguments include:

- **fitness_threshold**: Fitness threshold to stop the algorithm early.
- **init_genes**: Initial set of gene values for the initial population.
- **verbosity**: Level of verbosity (``0`` for silent, ``1`` for normal output, ``2`` for detailed output).

Examples
~~~~~~~~

Example 1: Numerical Genes
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, each individual represents a point in the xy-plane, and the fitness function is:

.. math::

   f(x, y) = -A \left( \sqrt{x^2 + y^2} - R \right)^2

which has an extended maximum on the circle centered at the origin with radius :math:`R`. The goal is to find points evenly distributed along the circle.

**Python Code:**

.. code-block:: python

   from lightweight_genetic_algorithm import GeneticAlgorithm

   # Define fitness function
   def fitness_function(individual):  # individual is x, y coordinates
       distance = (individual[0]**2 + individual[1]**2)**0.5
       R = 5  # Circle radius
       A = 5  # Overall fitness scaling
       fitness = -A * (distance - R)**2
       return fitness

   # Define ranges of the genes
   gene_ranges = [(-10, 10), (-10, 10)]

   # Create GeneticAlgorithm instance
   ga = GeneticAlgorithm(fitness_function, gene_ranges, crossover_method='Between')

   # Run the genetic algorithm
   all_populations = ga.run_light(n_generations=20, population_size=100)

After 20 generations, the individuals are evenly distributed along the circle.

.. image:: ../Figure_A2.png
   :alt: Initial and final populations for the numerical genes example.

Example 2: Categorical Genes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, each individual corresponds to an amino acid sequence composed of Lysine (K) and Glutamic acid (E). The goal is to find sequences with a sequence charge decoration (SCD) parameter near a given target value.

**Python Code:**

.. code-block:: python

   from lightweight_genetic_algorithm import GeneticAlgorithm

   # Calculate the sequence charge decoration (SCD) parameter
   def calculate_SCD(sequence):
       aa_charges = {'K': 1, 'E': -1}
       charge_sequence = [aa_charges[aa] for aa in sequence]
       SCD = sum(charge_sequence[a] * charge_sequence[b] * (b - a)**0.5
                 for a in range(len(charge_sequence) - 1)
                 for b in range(a + 1, len(charge_sequence)))
       SCD /= len(charge_sequence)
       return SCD

   # Define fitness function
   def fitness_function(sequence, target_SCD):
       SCD = calculate_SCD(sequence)
       fitness = -(SCD - target_SCD)**2
       return fitness

   # Main function
   def main():
       gene_ranges = ['E', 'K']  # Glutamic acid (E), Lysine (K)
       N = 50  # Sequence length
       target_SCD = -10  # Target SCD value

       ga = GeneticAlgorithm(
           fitness_function,
           gene_ranges,
           number_of_genes=N,
           fitness_function_args=(target_SCD,),
           use_multiprocessing=True
       )

       all_populations = ga.run_light(n_generations=50, population_size=100)

   if __name__ == '__main__':
       main()

After 50 generations, the final population consists of sequences with SCD values close to the target value.

.. image:: ../Figure_A4.png
   :alt: SCD values and net charges for initial and final populations.

Further Reading
---------------

For more detailed examples and advanced usage, refer to the documentation and source code available on `GitHub <https://github.com/JoseEliel/lightweight_genetic_algorithm>`_.
