"""
Implements an Ising model.
"""

import random

from markov import Clique, MarkovNetwork


class IsingModel(object):

    def __init__(self, rows, columns):

        # First, generate a list of K1 cliques
        cliques = [
            Clique([self.node_id(i, j)], self.random_k1_factor())
            for i in range(rows)
            for j in range(columns)
        ]

        # K2 cliques
        for i in range(rows):
            for j in range(columns):
                # To avoid double counting, consider edges from up to down and
                # from left to right

                # Not on the lower edge of the graph
                if i + 1 < rows:
                    cliques.append(Clique(
                        [self.node_id(i, j), self.node_id(i + 1, j)],
                        self.random_k2_factor()
                    ))
                if j + 1 < columns:
                    cliques.append(Clique(
                        [self.node_id(i, j), self.node_id(i, j + 1)],
                        self.random_k2_factor()
                    ))

    @staticmethod
    def node_id(row, column):
        """
        Generates a human-readable identifier for the node.
        """

        return f"{row}_{column}"

    @staticmethod
    def random_k1_factor():
        """
        Generates a random K1 factor function, with bias in the interval [-1, 1].
        """

        bias = 2 * random.random() - 1
        factor = {
            (True, ): bias,
            (False, ): -1 * bias
        }
        return factor

    @staticmethod
    def random_k2_factor():
        """
        Generates a random K2 factor function, with couplings in the interval [-1, 1].
        """

        coupling = 2 * random.random() - 1
        factor = {
            (True, True): coupling,
            (True, False): -1 * coupling,
            (False, True): -1 * coupling,
            (False, False): coupling,
        }
        return factor
