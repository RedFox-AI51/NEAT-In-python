# Phenotype.py
from .Node import Node, NodeType
from .Connection import Connection
from .Network import Network

# Import display libraries
import matplotlib.pyplot as plt
import networkx as nx


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
            NodeType.INPUT: 'lightgreen',   # Color for Input nodes
            NodeType.OUTPUT: 'lightcoral',  # Color for Output nodes
            NodeType.HIDDEN: 'lightblue'    # Color for Hidden nodes
        }

        # Add nodes to the graph with their colors
        for node in self.nodes:
            graph.add_node(node.id, label=node.ntype)

        # Add connections to the graph
        for connection in self.connections:
            # Check if the connection is enabled or disabled
            if connection.enabled:
                # For enabled connections, use solid lines
                graph.add_edge(connection.from_node.id, connection.to_node.id, weight=connection.weight, style='solid')
            else:
                # For disabled connections, use dotted red lines
                graph.add_edge(connection.from_node.id, connection.to_node.id, weight=connection.weight, style='dotted')

        # Use shell_layout for more organized positioning (concentric circles)
        pos = nx.shell_layout(graph)  # Use Shell Layout for better organization

        # Separate edges by style
        edges = graph.edges()
        solid_edges = [(u, v) for u, v in edges if graph[u][v]['style'] == 'solid']
        dotted_edges = [(u, v) for u, v in edges if graph[u][v]['style'] == 'dotted']

        # Draw nodes with respective colors based on node type
        node_colors_list = [node_colors[node.ntype] for node in self.nodes]
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color=node_colors_list, font_size=10)

        # Draw solid edges in black
        nx.draw_networkx_edges(graph, pos, edgelist=solid_edges, edge_color='black', style='solid')

        # Draw dotted edges in red
        nx.draw_networkx_edges(graph, pos, edgelist=dotted_edges, edge_color='red', style='dotted')

        # Draw edge labels (weights)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        # Show plot
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