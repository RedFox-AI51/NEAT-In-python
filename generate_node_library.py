import json
from NEAT.Activations import ActivationFunctions

# Get all activation functions
activation_functions = {name: func for name, func in vars(ActivationFunctions).items() if callable(func)}

# Create hidden nodes
hidden_nodes = [
    {
        "id": i + 10,  # Unique ID starting from 10
        "type": "HIDDEN",
        "value": 0.0,
        "activation": name  # Store activation function name
    }
    for i, (name, activation) in enumerate(activation_functions.items())
]

# Save to JSON file
json_filename = "hidden_nodes_library.json"
with open(json_filename, "w") as json_file:
    json.dump(hidden_nodes, json_file, indent=4)

print(f"Saved hidden nodes to {json_filename}")
