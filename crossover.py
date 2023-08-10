import numpy as np

class CrossoverBetween:
    """
    This class represents a crossover method where the child's genes are chosen 
    to be a random value between the corresponding genes of the two parents.
    """
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
            parent1 (list): The genes of the first parent.
            parent2 (list): The genes of the second parent.

        Returns:
            list: The genes of the child.
        """
        return [np.random.uniform(low=min(p1, p2), high=max(p1, p2)) for p1, p2 in zip(parent1, parent2)]


class CrossoverMidpoint:
    """
    This class represents a crossover method where the child's genes are chosen 
    to be the average of the corresponding genes of the two parents.
    """
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
            parent1 (list): The genes of the first parent.
            parent2 (list): The genes of the second parent.

        Returns:
            list: The genes of the child.
        """
        return [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]


class CrossoverEitherOr:
    """
    This class represents a crossover method where the child's genes are chosen 
    to be either the corresponding gene of the first parent or the second parent.
    """
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
            parent1 (list): The genes of the first parent.
            parent2 (list): The genes of the second parent.

        Returns:
            list: The genes of the child.
        """
        return [p1 if np.random.rand() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]