# Node.py
import random
from enum import Enum
from typing import Callable, Optional

class NodeType(Enum):
    INPUT = "Input"
    HIDDEN = "Hidden"
    OUTPUT = "Output"

class Node:
    def __init__(self, id: int, ntype: NodeType, value: float, activation: Optional[Callable[[float], float]] = None, bias=None):
        self.id = id
        self.ntype = ntype
        self.value = value
        self.activation = activation
        self.bias = bias

        # Bias should be included for both hidden and output nodes
        if self.bias == None:
            if ntype in {NodeType.HIDDEN, NodeType.OUTPUT}:
                self.bias = random.uniform(-1.0, 1.0)
            else:
                self.bias = 0.0  # Input nodes have no bias

    def get_output(self) -> float:
        """Compute the node's output with bias and activation."""
        output = self.value + self.bias
        return self.activation(output) if self.activation else output

    def copy(self):
        return Node(self.id, self.ntype, self.value, self.activation)
    
    def __repr__(self):
        return str({"id": self.id, "type": self.ntype.value, "value": self.value, "activation": self.activation, "bias": self.bias})

if __name__ == "__main__":
    # Example Usage
    import math

    relu = lambda x: max(0, x)

    node = Node(1, NodeType.HIDDEN, 0.5, relu)
    print(f"Node {node.id} Output: {node.get_output()}")  # Applies bias and ReLU
