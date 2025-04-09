# Crossover.py
from .Network import Network
from .Node import Node, NodeType
from .Connection import Connection
from .Activations import ActivationFunctions

import random

class Crossover:
    def __init__(self):
        pass

    def Crossover(self, parent_1:Network, parent_2:Network):
        if parent_1.fitness > parent_2.fitness:
            return self._crossover_instance_1(parent_1, parent_2)
        elif parent_2.fitness > parent_1.fitness:
            return self._crossover_instance_2(parent_1, parent_2)
        else:
            # If both parents have the same fitness, randomly choose one
            return random.choice([self._crossover_instance_1(parent_1, parent_2), self._crossover_instance_2(parent_1, parent_2)])

    def _crossover_instance_1(self, parent_1: Network, parent_2: Network):
        """
        This method performs crossover between two parent networks.
        It now uses `_find_disjoint_excess_genes()` to identify genes.

        Args:
            parent_1 (Network): The fitter parent.
            parent_2 (Network): The less fit parent.

        Returns:
            Network: A new child network from the crossover process.
        """

        # Step 1: Copy parent_1's nodes into child
        child_nodes = [node.copy() for node in parent_1.nodes]
        child_conns = []

        # Step 2: Find disjoint & excess genes
        disjoint, excess = self._find_disjoint_excess_genes(parent_1, parent_2)

        # Step 3: Iterate through connections and perform crossover
        i, j = 0, 0  # Pointers for parent_1 and parent_2 connection lists

        while i < len(parent_1.conns) or j < len(parent_2.conns):
            if i < len(parent_1.conns):
                conn_1 = parent_1.conns[i]
            else:
                conn_1 = None

            if j < len(parent_2.conns):
                conn_2 = parent_2.conns[j]
            else:
                conn_2 = None

            # Get current innovation number
            innov_1 = conn_1.Innov if conn_1 else float('inf')
            innov_2 = conn_2.Innov if conn_2 else float('inf')

            if innov_1 < innov_2:
                # conn_1 is disjoint or excess → inherit from fitter parent
                if innov_1 in disjoint or innov_1 in excess:
                    child_conns.append(conn_1.copy())
                i += 1
            elif innov_2 < innov_1:
                # conn_2 is disjoint or excess → skip since it's from less fit parent
                j += 1
            else:
                # Both parents have the gene → randomly pick one
                chosen_conn = random.choice([conn_1, conn_2]).copy()
                i += 1
                j += 1

                # Handle disabled connections (75% chance of staying disabled)
                if not conn_1.enabled or (conn_2 and not conn_2.enabled):
                    if random.random() < 0.75:
                        chosen_conn.enabled = False

                child_conns.append(chosen_conn)

        # Step 4: Create and return the new child network
        child = Network(nodes=child_nodes, conns=child_conns)
        return child

    def _crossover_instance_2(self, parent_1: Network, parent_2: Network):
        """
        This method performs crossover between two parent networks.
        This version assumes parent_2 is more fit than parent_1.

        Args:
            parent_1 (Network): The less fit parent.
            parent_2 (Network): The fitter parent.

        Returns:
            Network: A new child network from the crossover process.
        """

        # Step 1: Copy parent_2's nodes into child
        child_nodes = [node.copy() for node in parent_2.nodes]
        child_conns = []

        # Step 2: Find disjoint & excess genes
        disjoint, excess = self._find_disjoint_excess_genes(parent_2, parent_1)

        # Step 3: Iterate through connections and perform crossover
        i, j = 0, 0  # Pointers for parent_2 and parent_1 connection lists

        while i < len(parent_2.conns) or j < len(parent_1.conns):
            if i < len(parent_2.conns):
                conn_2 = parent_2.conns[i]
            else:
                conn_2 = None

            if j < len(parent_1.conns):
                conn_1 = parent_1.conns[j]
            else:
                conn_1 = None

            # Get current innovation number
            innov_2 = conn_2.Innov if conn_2 else float('inf')
            innov_1 = conn_1.Innov if conn_1 else float('inf')

            if innov_2 < innov_1:
                # conn_2 is disjoint or excess → inherit from fitter parent
                if innov_2 in disjoint or innov_2 in excess:
                    child_conns.append(conn_2.copy())
                i += 1
            elif innov_1 < innov_2:
                # conn_1 is disjoint or excess → skip since it's from less fit parent
                j += 1
            else:
                # Both parents have the gene → randomly pick one
                chosen_conn = random.choice([conn_1, conn_2]).copy()
                i += 1
                j += 1

                # Handle disabled connections (75% chance of staying disabled)
                if not conn_2.enabled or (conn_1 and not conn_1.enabled):
                    if random.random() < 0.75:
                        chosen_conn.enabled = False

                child_conns.append(chosen_conn)

        # Step 4: Create and return the new child network
        child = Network(nodes=child_nodes, conns=child_conns)
        return child


    def _find_disjoint_excess_genes(self, parent_1: Network, parent_2: Network):
        """
        This method finds the disjoint and excess genes between two parent networks.
        It will return a tuple containing two lists: disjoint and excess genes.

        Args:
            parent_1 (Network): This is the network of parent 1
            parent_2 (Network): This is the network of parent 2

        Returns:
            tuple: A tuple containing two lists: disjoint and excess genes.
        """

        disjoint = []
        excess = []

        # Get the innovation numbers of both parents
        innov_1 = sorted(conn.Innov for conn in parent_1.conns)
        innov_2 = sorted(conn.Innov for conn in parent_2.conns)

        # print(f"Parent 1 Innovations: {innov_1}")
        # print(f"Parent 2 Innovations: {innov_2}")

        max_innov_1 = max(innov_1, default=0)
        max_innov_2 = max(innov_2, default=0)
        max_common = min(max_innov_1, max_innov_2)

        # Identify disjoint and excess genes
        innov_set_1 = set(innov_1)
        innov_set_2 = set(innov_2)

        for innov in innov_set_1.symmetric_difference(innov_set_2):  
            if innov > max_common:
                excess.append(innov)
            else:
                disjoint.append(innov)
        
        # print(f"Disjoint Genes: {disjoint}")
        # print(f"Excess Genes: {excess}")

        return (disjoint, excess)
