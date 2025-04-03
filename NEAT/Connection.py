# Connection.py
from .Node import Node, NodeType

class Connection:
    def __init__(self, Innov: int, weight: float, from_node: Node, to_node: Node, enabled: bool = True):
        # Disallowed connections
        if from_node.ntype == NodeType.OUTPUT and to_node.ntype == NodeType.INPUT:
            raise ValueError("Cannot connect Output to Input")
        if from_node.ntype == NodeType.HIDDEN and to_node.ntype == NodeType.INPUT:
            raise ValueError("Cannot connect Hidden to Input")
        
        # Allowed connections:
        # Input -> Hidden, Input -> Output
        # Hidden -> Hidden, Hidden -> Output
        # Output -> Hidden (rare case)
        allowed = (
            (from_node.ntype == NodeType.INPUT and to_node.ntype in {NodeType.HIDDEN, NodeType.OUTPUT}) or
            (from_node.ntype == NodeType.HIDDEN and to_node.ntype in {NodeType.HIDDEN, NodeType.OUTPUT}) or
            (from_node.ntype == NodeType.OUTPUT and to_node.ntype == NodeType.HIDDEN)  # Rare case
        )

        if not allowed:
            raise ValueError(f"Invalid connection from {from_node.ntype.value} to {to_node.ntype.value}")

        self.Innov = Innov  # Innovation number for tracking evolution (for NEAT-like algorithms)
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled

    def forward(self):
        """Transfers the weighted value if the connection is enabled."""
        if self.enabled:
            weighted_input = self.from_node.get_output() * self.weight
            self.to_node.value += weighted_input
            return weighted_input  # Optional: return the weighted value transferred
        

    def __repr__(self):
        return f"Connection(Innov={self.Innov}, From={self.from_node.id}, To={self.to_node.id}, Weight={self.weight}, Enabled={self.enabled})"

    def get_weighted_input(self):
        """Returns the weighted input value before it's transferred."""
        return self.from_node.get_output() * self.weight

    def copy(self):
        return Connection(self.Innov, self.weight, self.from_node, self.to_node, self.enabled)