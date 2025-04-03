# NEAT-In-Python

## Overview
This repository implements the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm in Python. NEAT is a genetic algorithm designed to evolve artificial neural networks by optimizing both their connection weights and topologies over generations.

This implementation is based on the original paper by Kenneth O. Stanley, which is included in this repository as `stanley.ec02.pdf`.

## Features
- Custom activation functions
- Node and connection mutation
- Genetic crossover
- Phenotype expression of neural networks
- Sample neural network implementation

## Installation
Ensure you have Python installed on your system. Clone the repository and install dependencies (if any):

```sh
git clone https://github.com/RedFox-AI51/NEAT-In-python.git
cd NEAT-In-python
```

## File Structure
```
NEAT-In-python/
|── NEAT/
│──── Activations.py        # Defines activation functions
│──── Connection.py         # Manages network connections
│──── Crossover.py          # Handles genetic crossover
│──── Mutate.py             # Implements mutation operations
│──── Network.py            # Defines the neural network structure
│──── Node.py               # Manages individual nodes (neurons)
│──── Phenotype.py          # Converts genotype into a working neural network
│── SampleNetwork.py      # Example usage of NEAT
│── stanley.ec02.pdf      # Original NEAT research paper
```

## Usage
You can test the NEAT implementation using `SampleNetwork.py`:

```sh
python SampleNetwork.py
```

## Future Improvements
- Add support for different evolutionary strategies
- Improve efficiency with parallel processing
- Implement visualization tools for evolved networks

## References
- [Original NEAT Paper by Kenneth O. Stanley](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [Better NEAT Explination](https://macwha.medium.com/evolving-ais-using-a-neat-algorithm-2d154c623828)

## License
This project is open-source. Feel free to contribute or modify it as needed!

---

For any issues or contributions, please submit a pull request or open an issue on GitHub.
