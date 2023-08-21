import numpy as np
from .population import Individual, NumericGene

class Crossover:
    """
    This is a base class for all crossover methods.
    """
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.

        Returns:
            Individual: The child.
        """
        raise NotImplementedError("Subclass must implement this method")


class CrossoverBetween(Crossover):
    """
    This class represents a crossover method where the child's genes are chosen 
    to be a random value between the corresponding genes of the two parents.
    """
    def crossover(self, parent1, parent2):
        child_genes = []
        for g1, g2 in zip(parent1.get_genes(), parent2.get_genes()):
            if "between" not in g1.crossover_methods:
                raise ValueError(f"The crossover method 'between' is not compatible with the gene type.")
            low = min(g1.value, g2.value)
            high = max(g1.value, g2.value)
            value = np.random.uniform(low=low, high=high)
            child_genes.append(NumericGene(low, high, value))
        return Individual(child_genes)

class CrossoverMidpoint(Crossover):
    """
    This class represents a crossover method where the child's genes are chosen 
    to be the average of the corresponding genes of the two parents.
    """
    def crossover(self, parent1, parent2):
        child_genes = []
        for g1, g2 in zip(parent1.get_genes(), parent2.get_genes()):
            if "midpoint" not in g1.crossover_methods:
                raise ValueError(f"The crossover method 'midpoint' is not compatible with the gene type.")
            low = min(g1.low, g2.low)
            high = max(g1.high, g2.high)
            value = (g1.value + g2.value) / 2
            child_genes.append(NumericGene(low, high, value))
        return Individual(child_genes)

class CrossoverEitherOr(Crossover):
    """
    This class represents a crossover method where the child's genes are chosen 
    to be either the corresponding gene of the first parent or the second parent.
    """
    def crossover(self, parent1, parent2):
        child_genes = []
        for g1, g2 in zip(parent1.get_genes(), parent2.get_genes()):
            if "either or" not in g1.crossover_methods:
                raise ValueError(f"The crossover method 'either or' is not compatible with the gene type.")
            child_genes.append(g1 if np.random.rand() < 0.5 else g2)
        return Individual(child_genes)