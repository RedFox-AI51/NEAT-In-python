# Network.py
from .Node import Node, NodeType
from .Connection import Connection

class Network:
    def __init__(self, nodes: list[Node], conns: list[Connection]):
        for n in nodes:
            if not isinstance(n, Node):
                raise TypeError("nodes are not of type Node")
        for c in conns:
            if not isinstance(c, Connection):
                raise TypeError("conns are not of type Connection")
        
        self.nodes = nodes
        self.conns = conns
        self.fitness = 0.0 # Fitness score for the network
        self.processed_nodes = set()  # Keep track of processed nodes during input propagation
    
    def reset(self):
        """Reset the network by clearing the processed nodes set."""
        self.processed_nodes.clear()
        # Reset all node values to 0
        for node in self.nodes:
            node.value = 0

    def run(self):
        """Run the network, processing connections layer by layer using pass_input."""
        # Reset non-input nodes before propagation
        for node in self.nodes:
            if node.ntype != NodeType.INPUT:
                node.value = 0  # Reset to prevent residual values

        # Feed the input nodes using pass_input
        input_nodes = self.get_input_nodes()
        self.processed_nodes.clear()  # Reset the processed nodes set for each run
        
        for in_node in input_nodes:
            self.pass_input(in_node)

        # After propagating input, activate hidden and output nodes
        for node in self.nodes:
            if node.ntype != NodeType.INPUT:
                # Only update non-input nodes that have incoming connections
                if any(conn.to_node == node for conn in self.conns if conn.enabled):
                    node.value = node.get_output()  # Process activation if it has inputs
                else:
                    node.value = 0  # No input? Output stays 0

    def pass_input(self, in_node: Node):
        """Feed forward 1 node at a time, processing connections from an individual input node."""
        # If the node has already been processed, return immediately to prevent recursion
        if in_node in self.processed_nodes:
            return

        self.processed_nodes.add(in_node)  # Mark this node as processed

        conns = self.conns
        in_value = in_node.get_output()

        for conn in conns:
            if conn.from_node == in_node and conn.enabled:
                to_node = conn.to_node
                to_node.value += in_value * conn.weight  # Propagate input value

                # Only propagate further if the node is not an input node
                if to_node.ntype != NodeType.INPUT:
                    self.pass_input(to_node)  # Propagate further if necessary
    
    def get_output(self):
        """Return the output value of the output nodes."""
        return [node.get_output() for node in self.nodes if node.ntype == NodeType.OUTPUT]
    
    def get_input_nodes(self):
        """Return the input nodes."""
        return [node for node in self.nodes if node.ntype == NodeType.INPUT]
    
    def get_hidden_nodes(self):
        """Return the hidden nodes."""
        return [node for node in self.nodes if node.ntype == NodeType.HIDDEN]
    
    def get_output_nodes(self):
        """Return the output nodes."""
        return [node for node in self.nodes if node.ntype == NodeType.OUTPUT]
    
    def summary(self):
        return  "Nodes:\n" \
                f"    Input shape: {len(self.get_input_nodes())}\n" \
                f"    Output shape: {len(self.get_output_nodes())}\n" \
                f"    Number of hidden nodes: {len(self.get_hidden_nodes())}\n" \
                "Connections:\n" \
                f"{[conn.__repr__() for conn in self.conns]}"

    def copy(self):
        """Create a deep copy of the network."""
        copied_nodes = [node.copy() for node in self.nodes]
        copied_conns = [conn.copy() for conn in self.conns]
        return Network(copied_nodes, copied_conns)