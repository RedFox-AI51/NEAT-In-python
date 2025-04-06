from NEAT.NEAT import NEAT
from NEAT.Network import Network
from NEAT.Node import Node, NodeType
from NEAT.Connection import Connection
from NEAT.Activations import ActivationFunctions

import matplotlib.pyplot as plt

# Constants
POPULATION_SIZE = 10  # Size of the population
GENERATIONS = 100  # Number of generations to evolve
MUTATION_RATE = 0.1  # Mutation rate for the NEAT algorithm

# Define the nodes in the preferred order: input, output, hidden
nodes = [
    Node(1, NodeType.INPUT, 0.0),   # Input node 1
    Node(2, NodeType.INPUT, 0.0),   # Input node 2
    Node(3, NodeType.OUTPUT, 0.0),  # Output node
    Node(4, NodeType.HIDDEN, 0.0)   # Hidden node
]

# Define the connections (from inputs to hidden, hidden to output)
conns = [
    Connection(Innov=1, weight=0.5, from_node=nodes[0], to_node=nodes[3], enabled=True),  # Input 1 -> Hidden
    Connection(Innov=2, weight=0.5, from_node=nodes[1], to_node=nodes[3], enabled=True),  # Input 2 -> Hidden
    Connection(Innov=3, weight=0.8, from_node=nodes[3], to_node=nodes[2], enabled=True)   # Hidden -> Output
]

# Fitness function for the XOR problem
def fitness_function(network: Network) -> float:
    inputs = [
        (0, 0),  # Expected output: 0
        (0, 1),  # Expected output: 1
        (1, 0),  # Expected output: 1
        (1, 1)   # Expected output: 0
    ]
    expected_outputs = [0, 1, 1, 0]
    fitness = 0.0

    for node in network.nodes:
        node.value = 0.0

    for i, (input_1, input_2) in enumerate(inputs):
        network.nodes[0].value = input_1
        network.nodes[1].value = input_2
        network.run()
        output = network.get_output()[0]
        fitness += (expected_outputs[i] - output) ** 2

    return -fitness  # Negative because we want to minimize error

# Create initial population
def create_population(pop_size: int) -> list:
    population = []
    for _ in range(pop_size):
        new_network = Network(nodes, conns)
        genome = NEAT(new_network)
        population.append(genome)
    return population

# Main evolution loop
genomes: list[NEAT] = create_population(POPULATION_SIZE)
best_fitnesses = []
avg_fitnesses = []

for gen in range(GENERATIONS):
    gen_fitnesses = []
    for genome in genomes:
        fitness = fitness_function(genome.network)
        gen_fitnesses.append(fitness)
        genome.mutate(MUTATION_RATE)

    avg_fitness = sum(gen_fitnesses) / len(gen_fitnesses)
    best_fitness = max(gen_fitnesses)

    avg_fitnesses.append(avg_fitness)
    best_fitnesses.append(best_fitness)

    best_genome = genomes[gen_fitnesses.index(best_fitness)]
    print(f"Generation {gen}: Avg Fitness = {avg_fitness:.4f}, Best Fitness = {best_fitness:.4f}")

# Final result
print("\n=== Final Network ===")
# best_genome.phenotype.visualize()
# print(best_genome.network.summary())

# Plotting fitness over generations
plt.figure(figsize=(10, 5))
plt.plot(avg_fitnesses, label="Average Fitness", color="blue")
plt.plot(best_fitnesses, label="Best Fitness", color="green")
plt.title("Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
