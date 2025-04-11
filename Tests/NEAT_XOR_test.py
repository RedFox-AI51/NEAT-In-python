import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from NEAT.Genome import Genome
from NEAT.Network import Network
from NEAT.Node import Node, NodeType
from NEAT.Connection import Connection
from NEAT.Activations import ActivationFunctions
from NEAT.Crossover import Crossover

import json

def save_family_tree_to_file(family_tree, filename="family_tree.json"):
    with open(filename, "w") as file:
        json.dump(family_tree, file, indent=4)

# Constants
POPULATION_SIZE = 100              # Size of the population
GENERATIONS = 100                  # Number of generations to evolve
MUTATION_RATE = 0.3                # Mutation rate for the NEAT algorithm
ELITE_COUNT = 15                   # Number of elite genomes to carry over to the next generation

family_tree = {} # Dictionary to store family tree information

# Define the nodes in the preferred order: input, output, hidden
nodes = [
    Node(1, NodeType.INPUT, 0.0),   # Input node 1
    Node(2, NodeType.INPUT, 0.0),   # Input node 2
    Node(3, NodeType.OUTPUT, 0.0, ActivationFunctions.TanH),  # Output node
    Node(4, NodeType.HIDDEN, 0.0, ActivationFunctions.TanH),  # Hidden node
]

# Define the connections (from inputs to hidden, hidden to output)
conns = [
    Connection(Innov=1, weight=0.5, from_node=nodes[0], to_node=nodes[3], enabled=True),  # Input 1 to Hidden
    Connection(Innov=2, weight=0.5, from_node=nodes[1], to_node=nodes[3], enabled=True),  # Input 2 to Hidden
    Connection(Innov=3, weight=0.8, from_node=nodes[3], to_node=nodes[2], enabled=True),  # Hidden to Output
]

cross = Crossover()

# Fitness function for the XOR problem
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

# Create initial population
def create_population(pop_size: int) -> list:
    population = []
    for _ in range(pop_size):
        new_network = Network(nodes, conns)
        genome = Genome(new_network)
        population.append(genome)
        family_tree[genome.id] = {"parents": [None, None], "generation": 0}
    return population

def select_parents(population):
    """
    Selects two parents from the population using fitness-proportionate selection.
    Returns a tuple of (parent1, parent2).
    """
    # Normalize fitness values (shift if necessary to handle negative fitness)
    min_fitness = min(genome.network.fitness for genome in population)
    shift = 0
    if min_fitness < 0:
        shift = -min_fitness + 1e-6  # Small constant to avoid zero fitness

    total_fitness = sum(genome.network.fitness + shift for genome in population)
    
    def weighted_random_choice():
        r = random.uniform(0, total_fitness)
        cumulative = 0
        for genome in population:
            cumulative += genome.network.fitness + shift
            if cumulative >= r:
                return genome

    parent1 = weighted_random_choice()
    parent2 = weighted_random_choice()

    # Optional: ensure they are not the same (if population is small)
    if parent1 is parent2:
        parent2 = weighted_random_choice()

    return parent1, parent2

def reproduce(genomes: list[Genome]) -> list[Genome]:
    # 1. Sort population by fitness (higher is better)
    sorted_population = sorted(genomes, key=lambda g: g.network.fitness, reverse=True)

    # 2. Select elites (unchanged top performers)
    elites = [genome.copy() for genome in sorted_population[:ELITE_COUNT]]

    # 3. Initialize next generation with elite genomes
    next_generation = elites.copy()

    # 4. Fill the rest of the population
    while len(next_generation) < POPULATION_SIZE:
        parent1, parent2 = select_parents(genomes)

        net1 = parent1.network.copy()
        net2 = parent2.network.copy()

        # Crossover returns a new child genome
        child_net = cross.Crossover(net1, net2)
        child = Genome(child_net)

        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parent_ids = [parent1.id, parent2.id]

        family_tree[child.id] = {"parents": [parent1.id, parent2.id], "generation": child.generation}

        # Apply mutation to the child
        child.mutate(MUTATION_RATE)

        next_generation.append(child)

    return next_generation


# Main evolution loop
genomes: list[Genome] = create_population(POPULATION_SIZE)
best_fitnesses = []
avg_fitnesses = []

solved_genome = None

genomes[0].phenotype.visualize()

for gen in range(GENERATIONS):

    gen_fitnesses = []

    for genome in genomes:
        fitness = fitness_function(genome.network)
        genome.network.fitness = fitness  # Add this line
        gen_fitnesses.append(fitness)
        if solved_genome == None or genome.network.fitness > solved_genome.network.fitness:
            solved_genome = genome

    if solved_genome.network.fitness == -1.0:
        print("Solution found!")
        break

    avg_fitness = sum(gen_fitnesses) / len(gen_fitnesses)
    best_fitness = max(gen_fitnesses)

    avg_fitnesses.append(avg_fitness)
    best_fitnesses.append(best_fitness)

    best_genome = genomes[gen_fitnesses.index(best_fitness)]
    print(f"Generation {gen}: Avg Fitness = {avg_fitness:.4f}, Best Fitness = {best_fitness:.4f}")

    genomes = reproduce(genomes)
    

    # Reset the fitnesses of the genomes
    for genome in genomes:
        genome.network.fitness = 0.0
    
        
    # Shuffle the genomes
    # random.shuffle(genomes)


# Save the fitness data to a text file
output_path = "fitness_summary.txt"
with open(output_path, "w") as f:
    f.write("gen | avg | best\n")
    for gen in range(GENERATIONS):
        f.write(f"{gen} | {avg_fitnesses[gen]:.4f} | {best_fitnesses[gen]:.4f}\n")

print(f"Fitness summary saved to {output_path}")

# Save the best genome to a file
best_genome_path = "best_genome.txt"
with open(best_genome_path, "w") as f:
    f.write(str(best_genome.network.summary()))

# Create a matplotlib plot for saving as PNG
plt.figure(figsize=(10, 6))
plt.plot(range(GENERATIONS), avg_fitnesses, label='Avg Fitness', linestyle='--')
plt.plot(range(GENERATIONS), best_fitnesses, label='Best Fitness', linestyle='-')
plt.title('Fitness Comparison Between Iterations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PNG
plt.savefig('XOR_test1.png')
print("Fitness plot saved as XOR_test1.png")

# Optional: close the plot if not displaying interactively
plt.close()

# Create interactive plot using plotly
fig = go.Figure()

# Add traces (Avg Fitness and Best Fitness)
fig.add_trace(go.Scatter(
    x=list(range(GENERATIONS)),
    y=avg_fitnesses,
    mode='lines',
    name="Avg Fitness Iteration",
    line=dict(dash='dash')
))

fig.add_trace(go.Scatter(
    x=list(range(GENERATIONS)),
    y=best_fitnesses,
    mode='lines',
    name="Best Fitness Iteration",
    line=dict(dash='solid')
))

# Update layout for the plot
fig.update_layout(
    title="Fitness Comparison Between Iterations",
    xaxis_title="Generation",
    yaxis_title="Fitness",
    legend_title="Fitness Type",
    template="plotly_dark"
)

save_family_tree_to_file(family_tree, "family_tree.json")

# Show the interactive plot
fig.show()

best_genome.phenotype.visualize()