# SampleNetwork.py
from Node import Node, NodeType
from Connection import Connection
from Activations import ActivationFunctions

from Network import Network
from Mutate import Mutate

from Phenotype import Phenotype

import random


# Initialize nodes
nodes = [
    Node(1, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 1
    Node(2, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 2
    Node(3, NodeType.INPUT, round(random.uniform(-1.0, 1.0), 2)),  # Input Node 3
    Node(4, NodeType.OUTPUT, 0.0, ActivationFunctions.Sigmoid),  # Output Node 1
    Node(5, NodeType.HIDDEN, round(random.uniform(-1.0, 1.0), 2), ActivationFunctions.ReLu)  # Hidden Node with ReLU
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

network = Network(nodes, conns)
mutator = Mutate(network)
phenotype_1 = Phenotype(mutator.network)
phenotype_1.visualize()

# Process connections in correct order
for conn in network.conns:
    conn.forward()

# Print final values
for node in network.nodes:
    print(f"Node {node.id} ({node.ntype.value}): {node.value}")

print("\n")

mutator = Mutate(network)

# Mutate the network
for i in range(5):
    if random.randint(0,1):
        mutator.mutate_add_connection()
    else:
        mutator.mutate_add_node()

print("\n")

# Process connections in correct order (After Mutation)
for conn in network.conns:
    conn.forward()

# Print final values (After Mutation)
for node in network.nodes:
    print(f"Node {node.id} ({node.ntype.value}): {node.value}")

phenotype_2 = Phenotype(mutator.network)
phenotype_2.visualize()