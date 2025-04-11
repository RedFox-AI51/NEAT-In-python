from .Network import Network
from .Node import Node, NodeType
from .Connection import Connection
from .Phenotype import Phenotype
from .Mutate import Mutate

import uuid

class Genome:
    def __init__(self, network: Network):
        self.id = str(uuid.uuid4())
        self.network = network
        self.parent_ids = []
        self.generation = 0
        self.mutator = Mutate(self.network)
        self.phenotype = Phenotype(self.network)

    def random_mutation(self):
        # Apply random mutation to the network
        self.mutator.random_mutation()
    
    def mutate(self, mutation_rate: float = 0.1):
        # Apply mutation to the network
        self.mutator.mutate(mutation_rate)
    
    def copy(self):
        # Create a deep copy of the NEAT instance
        return Genome(self.network.copy())