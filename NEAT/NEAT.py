from .Network import Network
from .Node import Node, NodeType
from .Connection import Connection
from .Phenotype import Phenotype
from .Mutate import Mutate

class NEAT:
    def __init__(self, network: Network):
        self.network = network
        self.mutator = Mutate(self.network)
        self.phenotype = Phenotype(self.network)

    def random_mutation(self):
        # Apply random mutation to the network
        self.mutator.random_mutation()
    
    def mutate(self, mutation_rate: float = 0.1):
        # Apply mutation to the network
        self.mutator.mutate(mutation_rate)
