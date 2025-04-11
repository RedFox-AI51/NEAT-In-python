# Mutate.py
from .Network import Network
from .Activations import ActivationFunctions
from .InnovationTracker import tracker
from .Connection import Connection
from .Node import Node, NodeType

import random

class Mutate:
    def __init__(self, network: Network):
        if not isinstance(network, Network):
            raise TypeError("network is not of type Network")
        self.network = network
    
    def random_mutation(self):
        mutation_type = random.choice([
            self.mutate_add_connection,
            self.mutate_add_node,
            self.mutate_weights
        ])
        mutation_type()
    
    def mutate(self, mutation_rate: float = 0.1):
        if random.random() < mutation_rate:
            if random.randint(0,2) == 0:
                self.mutate_add_connection()
            elif random.randint(0,2) == 1:
                self.mutate_add_node()
            else:
                self.mutate_weights()
    
    def mutate_add_connection(self):
        nodes = self.network.nodes
        conns = self.network.conns

        max_attempts = 10  # Avoid infinite loops when finding valid connections

        for _ in range(max_attempts):
            from_node = random.choice(nodes)
            to_node = random.choice(nodes)

            # Ensure a valid connection based on rules:
            valid_connection = (
                (from_node.ntype == NodeType.INPUT and to_node.ntype in {NodeType.HIDDEN, NodeType.OUTPUT}) or
                (from_node.ntype == NodeType.HIDDEN and to_node.ntype in {NodeType.HIDDEN, NodeType.OUTPUT}) or
                (from_node.ntype == NodeType.OUTPUT and to_node.ntype == NodeType.HIDDEN)  # Rare case
            )

            # Prevent self-connections
            if from_node == to_node:
                continue

            # Check if the connection already exists
            if any(conn.from_node == from_node and conn.to_node == to_node for conn in conns):
                continue

            if valid_connection:
                # Use the innovation tracker to get a consistent innovation number
                innov = tracker.get_innovation(from_node.id, to_node.id)
                new_conn = Connection(
                    Innov=innov,
                    weight=round(random.uniform(-1.0, 1.0), 2),
                    from_node=from_node,
                    to_node=to_node,
                    enabled=True
                )
                self.network.conns.append(new_conn)
                return  # Exit after adding one connection

        # print("Failed to find a valid connection to add.")

    def mutate_add_node(self):
        conns = self.network.conns
        nodes = self.network.nodes

        if not conns:
            # print("No connections to mutate.")
            return

        # Pick a random connection to split
        conn = random.choice(conns)
        from_node = conn.from_node
        to_node = conn.to_node

        conn.enabled = False # Deactivate the connection

        # Create a new hidden node
        new_node_id = len(nodes) + 1
        new_node = Node(new_node_id, NodeType.HIDDEN, random.uniform(-1.0, 1.0), ActivationFunctions.ReLu)  # ReLU activation
        
        # Add the new node to the network
        self.network.nodes.append(new_node)

        # Create two new connections:
        # - from `from_node` to the new node
        # - from the new node to `to_node`
        weight_1 = round(random.uniform(-1.0, 1.0), 2)
        weight_2 = conn.weight
        innov_1 = len(self.network.conns) + 1
        innov_2 = innov_1 + 1
        
        new_conn_1 = Connection(innov_1, weight_1, from_node, new_node)
        new_conn_2 = Connection(innov_2, weight_2, new_node, to_node)

        self.network.conns.append(new_conn_1)
        self.network.conns.append(new_conn_2)

        # print(f"Added node {new_node.id} and split connection {from_node.id} → {to_node.id}")

    def mutate_weights(self):
        for conn in self.network.conns:
            if conn.enabled:
                if random.random() < 0.1:
                    conn.weight = round(random.uniform(-2.0, 2.0), 2)