import random

from NEAT.Node import Node, NodeType
from NEAT.Connection import Connection
from NEAT.Network import Network
from NEAT.Phenotype import Phenotype

# Define ReLU activation function
relu = lambda x: max(0, x)

# Initialize nodes
nodes = [
    Node(1, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 1
    Node(2, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 2
    Node(3, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 3
    Node(4, NodeType.OUTPUT, 0.0),  # Output Node
    Node(5, NodeType.HIDDEN, round(random.uniform(-1.0, 1.0), 2), relu)  # Hidden Node with ReLU
]

# Initialize connections
conns = [
    Connection(1, 0.7, nodes[0], nodes[3], True),  # Node 1 → Node 3
    Connection(2, -0.5, nodes[1], nodes[3], False),  # Node 2 → Node 3 (Disabled)
    Connection(3, 0.5, nodes[2], nodes[3], True),  # Node 3 → Node 3
    Connection(4, 0.2, nodes[1], nodes[4], True),  # Node 2 → Node 4
    Connection(5, 0.4, nodes[4], nodes[3], True),  # Node 4 → Node 3
    Connection(6, 0.6, nodes[0], nodes[4], True),  # Node 1 → Node 4
    Connection(11, 0.6, nodes[3], nodes[4], True),  # Node 3 → Node 4
]

# Create a Phenotype instance
phenotype = Phenotype(Network(nodes, conns))

# Visualize the phenotype
phenotype.visualize()