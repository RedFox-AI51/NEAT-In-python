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