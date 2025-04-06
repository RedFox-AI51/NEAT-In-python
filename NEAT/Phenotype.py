# Phenotype.py
from .Node import Node, NodeType
from .Connection import Connection
from .Network import Network

# Import display libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Phenotype:
    def __init__(self, network:Network):
        # Initialize with nodes and connections
        self.nodes = network.nodes if network.nodes else []
        self.connections = network.conns if network.conns else []

    def visualize(self):
        # Create a directed graph
        graph = nx.DiGraph()

        # Assign a color for each node type
        node_colors = {
            NodeType.INPUT: 'lightgreen',   # Input nodes
            NodeType.OUTPUT: 'lightcoral',  # Output nodes
            NodeType.HIDDEN: 'lightblue'    # Hidden nodes
        }

        # Group nodes by type
        input_nodes = [node for node in self.nodes if node.ntype == NodeType.INPUT]
        hidden_nodes = [node for node in self.nodes if node.ntype == NodeType.HIDDEN]
        output_nodes = [node for node in self.nodes if node.ntype == NodeType.OUTPUT]

        # Create node positions
        pos = {}

        # Function to space nodes vertically
        def place_nodes(layer_nodes, x_pos):
            """Places nodes in a vertical column at the specified x position."""
            n = len(layer_nodes)
            if n == 1:
                pos[layer_nodes[0].id] = (x_pos, 0)  # Center single node
            else:
                for i, node in enumerate(layer_nodes):
                    y_pos = -i + (n - 1) / 2  # Center nodes vertically
                    pos[node.id] = (x_pos, y_pos)

        # Position input and output nodes
        place_nodes(input_nodes, x_pos=0)     # Input layer (left)
        place_nodes(output_nodes, x_pos=4)    # Output layer (right)

        # Spread hidden nodes dynamically between inputs and outputs
        if hidden_nodes:
            num_hidden = len(hidden_nodes)
            x_positions = np.linspace(1, 3, num_hidden)  # Spread hidden nodes between x=1 and x=3
            for node, x in zip(hidden_nodes, x_positions):
                pos[node.id] = (x, np.random.uniform(-len(hidden_nodes)/2, len(hidden_nodes)/2))  # Randomized Y position

        # Add nodes to the graph
        for node in self.nodes:
            graph.add_node(node.id, label=node.ntype)

        # Add connections to the graph
        for connection in self.connections:
            if connection.enabled:
                graph.add_edge(connection.from_node.id, connection.to_node.id, weight=connection.weight, style=('solid_g' if connection.weight > 0 else 'solid_r'))
            else:
                graph.add_edge(connection.from_node.id, connection.to_node.id, weight=connection.weight, style='dotted')

        # Separate edges by style
        edges = graph.edges()
        solid_edges_g = [(u, v) for u, v in edges if graph[u][v]['style'] == 'solid_g']
        solid_edges_r = [(u, v) for u, v in edges if graph[u][v]['style'] == 'solid_r']
        dotted_edges = [(u, v) for u, v in edges if graph[u][v]['style'] == 'dotted']

        # Draw nodes with respective colors
        node_colors_list = [node_colors[node.ntype] for node in self.nodes]
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color=node_colors_list, font_size=10)

        # Draw solid edges in green and red
        nx.draw_networkx_edges(graph, pos, edgelist=solid_edges_g, edge_color='lime', style='solid')
        nx.draw_networkx_edges(graph, pos, edgelist=solid_edges_r, edge_color='red', style='solid')

        # Draw dotted edges in red
        nx.draw_networkx_edges(graph, pos, edgelist=dotted_edges, edge_color='lightgray', style='dotted')

        # Draw edge labels (weights)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        # Show plot
        plt.title("Neural Network Visualization")
        plt.show()


if __name__ == "__main__":
    import random

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
        Connection(1, round(random.uniform(-2.0, 2.0), 2), nodes[0], nodes[3], True),  # Node 1 → Node 3
        Connection(2, round(random.uniform(-2.0, 2.0), 2), nodes[1], nodes[3], False),  # Node 2 → Node 3 (Disabled)
        Connection(3, round(random.uniform(-2.0, 2.0), 2), nodes[2], nodes[3], True),  # Node 3 → Node 3
        Connection(4, round(random.uniform(-2.0, 2.0), 2), nodes[1], nodes[4], True),  # Node 2 → Node 4
        Connection(5, round(random.uniform(-2.0, 2.0), 2), nodes[4], nodes[3], True),  # Node 4 → Node 3
        Connection(6, round(random.uniform(-2.0, 2.0), 2), nodes[0], nodes[4], True),  # Node 1 → Node 4
        Connection(11, round(random.uniform(-2.0, 2.0), 2), nodes[3], nodes[4], True),  # Node 3 → Node 4
    ]

    # Create a Phenotype instance
    phenotype = Phenotype(Network(nodes, conns))

    # Visualize the phenotype
    phenotype.visualize()