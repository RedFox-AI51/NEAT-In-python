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
        self.fitness = 0

    def run(self):
        """Run the network, processing connections and updating node values."""
        # Process connections
        for conn in self.conns:
            conn.forward()
    
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