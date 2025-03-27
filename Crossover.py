# Crossover.py
from Network import Network

class Crossover:
    def __init__(self):
        pass

    def Crossover(self, parent_1:Network, parent_2:Network):
        if parent_1.fitness > parent_2.fitness:
            print("parent 2 cross")
        elif parent_2.fitness > parent_1.fitness:
            print("parent 1 cross")