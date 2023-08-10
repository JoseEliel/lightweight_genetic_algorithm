import numpy as np

class CrossoverBetween:
    def crossover(self, parent1, parent2):
        return [np.random.uniform(low=min(p1, p2), high=max(p1, p2)) for p1, p2 in zip(parent1, parent2)]

class CrossoverMidpoint:
    def crossover(self, parent1, parent2):
        return [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]

class CrossoverEitherOr:
    def crossover(self, parent1, parent2):
        return [p1 if np.random.rand() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]