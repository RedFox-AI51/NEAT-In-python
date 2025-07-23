import json
from NEAT import *

with open("Outputs\\best_genome.json") as f:
    data:dict = json.load(f)

saved_nodes:dict = data["nodes"]
saved_connections:dict = data["connections"]

network_nodes = []
network_connections = []

# Extract Saved Nodes
for node in saved_nodes.items():
    if node[0] == 'input':
        inputs = node
    if node[0] == 'output':
        outputs = node
    if node[0] == 'hidden':
        hidden = node

def GetActivation(name: str):
    print(name)
    name = name.strip("<").strip(">")
    func_data = name.split(".")
    if func_data[0] == 'None':
        return None
    func_data[1] = func_data[1].replace("at", "|").split("|")
    func = func_data[1][0]
    print(func, "\n")
    return {
        "Sigmoid": ActivationFunctions.Sigmoid,
        "ReLu": ActivationFunctions.ReLu,
        "TanHat": ActivationFunctions.TanH,
        "Gaussian": ActivationFunctions.Gaussian,
        "Linear": ActivationFunctions.Linear,
        "Sine": ActivationFunctions.Sine
    }.get(func, ActivationFunctions.Sigmoid)  # default to Sigmoid if not found

# Input Nodes
for input_node in inputs[1]:
    input_node:str = input_node
    refined_data = []
    input_node = input_node.strip("{").strip("}").replace(" ", "")
    refined_data = input_node.split(",")

    data_pairs:list[tuple] = []

    for ref_data in refined_data:
        label, d_value = ref_data.split(":")
        data_pairs.append((label, d_value))

    network_nodes.append(Node(int(data_pairs[0][1]), NodeType.INPUT, float(data_pairs[2][1]), GetActivation(data_pairs[3][1]), float(data_pairs[4][1])))

# Output Nodes
for output_node in outputs[1]:
    output_node:str = output_node
    refined_data = []
    output_node = output_node.strip("{").strip("}").replace(" ", "")
    refined_data = output_node.split(",")

    data_pairs:list[tuple] = []

    for ref_data in refined_data:
        label, d_value = ref_data.split(":")
        data_pairs.append((label, d_value))

    network_nodes.append(Node(int(data_pairs[0][1]), NodeType.OUTPUT, float(data_pairs[2][1]), GetActivation(data_pairs[3][1]), float(data_pairs[4][1])))

# Hidden Nodes
for hidden_node in hidden[1]:
    hidden_node:str = hidden_node
    refined_data = []
    hidden_node = hidden_node.strip("{").strip("}").replace(" ", "")
    refined_data = hidden_node.split(",")

    data_pairs:list[tuple] = []

    for ref_data in refined_data:
        label, d_value = ref_data.split(":")
        data_pairs.append((label, d_value))

    network_nodes.append(Node(int(data_pairs[0][1]), NodeType.HIDDEN, float(data_pairs[2][1]), GetActivation(data_pairs[3][1]), float(data_pairs[4][1])))

# print(network_nodes)

DEBUG = False

def SearchByID(nodes: list[Node], node_id):
    for node in nodes:
        if DEBUG:
            print(f"Node ID: {node.id} ( {type(node.id)} ) | Requested ID: {node_id} ( {type(node_id)} )")
        if node_id == node.id:
            return node
    raise ValueError(f"Node with ID {node_id} not found.")

# Extract Saved Connections
for connection in saved_connections:
    inovation = connection["innovation"]
    weight = connection["weight"]
    n_from = connection["from"]
    n_to = connection["to"]
    enabled = connection["enabled"]

    from_node = SearchByID(network_nodes, n_from)
    to_node = SearchByID(network_nodes, n_to)

    network_connections.append(Connection(int(inovation), float(weight), from_node, to_node, bool(enabled)))

# print(network_connections)

# Compile the Network
net = Network(network_nodes, network_connections)
print(net.summary())

# Build the Genome
genome = Genome(net)

# Fitness function for the XOR problem to Re-Test the genome
def fitness_function(network: Network) -> float:
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected_outputs = [0, 1, 1, 0]
    fitness = 0.0

    for node in network.nodes:
        node.value = 0.0

    for i, (input_1, input_2) in enumerate(inputs):
        network.reset()  # Reset the network before each run
        network.nodes[0].value = input_1
        network.nodes[1].value = input_2
        network.run()
        output = network.get_output()[0]
        fitness += (expected_outputs[i] - output) ** 2 # Squared error
    return -fitness  # Negative fitness for minimization

# Show saved fitness
saved_fitness = data["fitness"]
print(f"Saved fitness: {saved_fitness} | {100 - round(abs(saved_fitness), 2)} percent accuracy")

# Re-Test genome and compare fitnesses
genome.network.fitness = fitness_function(genome.network)
print(f"Fitness test: {genome.network.fitness} | {100 - round(abs(saved_fitness), 2)} percent accuracy")

fitness_difference = round(abs(genome.network.fitness), 2) - round(abs(saved_fitness), 2)

print(f"Fitness Difference: {round(abs(fitness_difference), 2)} percent")

genome.phenotype.visualize()