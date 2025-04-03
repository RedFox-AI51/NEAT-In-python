from NEAT.Network import Network
from NEAT.Node import Node, NodeType
from NEAT.Connection import Connection
from NEAT.Activations import ActivationFunctions
from NEAT.Crossover import Crossover
from NEAT.Phenotype import Phenotype

import random


cross = Crossover()
# For Network 1
nodes1 = [
    Node(1, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 1
    Node(2, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 2
    Node(3, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 3
    Node(4, NodeType.OUTPUT, 0.5, ActivationFunctions.Sigmoid),  # Output Node 1
    Node(5, NodeType.HIDDEN, round(random.uniform(-1.0, 1.0), 2), ActivationFunctions.ReLu),  # Hidden Node with ReLU
]

conns1 = [
    Connection(1, 0.7, nodes1[0], nodes1[3], True),  # Node 1 → Node 4
    Connection(2, -0.5, nodes1[1], nodes1[3], False),  # Node 2 → Node 4 (Disabled)
    Connection(3, 0.5, nodes1[2], nodes1[3], True),  # Node 3 → Node 4
    Connection(4, 0.2, nodes1[1], nodes1[4], True),  # Node 2 → Node 5
    Connection(5, 0.4, nodes1[4], nodes1[3], True),  # Node 5 → Node 4
    Connection(8, 0.6, nodes1[0], nodes1[4], True),  # Node 1 → Node 5
]

# For Network 2
nodes2 = [
    Node(1, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 1
    Node(2, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 2
    Node(3, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 3
    Node(4, NodeType.OUTPUT, 0.5, ActivationFunctions.Sigmoid),  # Output Node 1
    Node(5, NodeType.HIDDEN, round(random.uniform(-1.0, 1.0), 2), ActivationFunctions.ReLu),  # Hidden Node with ReLU
    Node(6, NodeType.HIDDEN, round(random.uniform(-1.0, 1.0), 2), ActivationFunctions.ReLu),  # Hidden Node with ReLU
]

conns2 = [
    Connection(1, 0.7, nodes2[0], nodes2[3], True),    # Node 1 → Node 4
    Connection(2, -0.5, nodes2[1], nodes2[3], False),  # Node 2 → Node 4 (Disabled)
    Connection(3, 0.5, nodes2[2], nodes2[3], True),    # Node 3 → Node 4
    Connection(4, 0.2, nodes2[1], nodes2[4], True),    # Node 2 → Node 5
    Connection(5, 0.4, nodes2[4], nodes2[3], False),   # Node 5 → Node 4
    Connection(6, 0.4, nodes2[4], nodes2[5], True),    # Node 5 → Node 6
    Connection(7, 0.4, nodes2[5], nodes2[3], True),    # Node 6 → Node 4
    Connection(9, 0.6, nodes2[2], nodes2[4], True),    # Node 3 → Node 5
    Connection(10, 0.6, nodes2[0], nodes2[5], True),   # Node 1 → Node 6
]

net1 = Network(nodes1, conns1)
net2 = Network(nodes2, conns2)

# Set Fitnesses
net1.fitness=round(random.uniform(-1.0, 1.0), 2)
net2.fitness=round(random.uniform(-1.0, 1.0), 2)

net3 = cross.Crossover(net1, net2)

Phenotype(net1).visualize()
Phenotype(net2).visualize()
Phenotype(net3).visualize()