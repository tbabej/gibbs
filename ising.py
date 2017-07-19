"""
Implements an Ising model.
"""

import random

from markov import Clique, MarkovNetwork


class IsingModel(object):

    def __init__(self, rows, columns):

        # First, generate a list of nodes
        cliques = []

        # K1 cliques
        for i in range(rows):
            for j in range(columns):
                # Random bias between -1 and 1
                bias = 2 * random.random() - 1
                factor = {
                    (True, ): bias,
                    (False, ): -1 * bias
                }
                cliques.append(Clique([(i, j)], factor))

        # K2 cliques
        for i in range(rows):
            for j in range(columns):
                # Random bias between -1 and 1
                coupling = 2 * random.random() - 1
                factor = {
                    (True, True): coupling,
                    (True, False): -1 * coupling,
                    (False, True): -1 * coupling,
                    (False, False): coupling,
                }
                cliques.append(Clique([(i, j)], factor))
