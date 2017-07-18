"""
This module implements functionality related to Markov Networks.
"""

import operator
from functools import reduce


class Clique(object):

    def __init__(self, nodes, potential):
        """
        Represents a clique in the Markov network.
        Takes:
          nodes - list of names of random variables
          potential - value of factor function for each assignment of random
                      values in the clique

        Example:
        >>> c = Clique(['P1', 'P2'],{
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        })
        """

        self.nodes = nodes
        self.potential = potential

    def factor(self, **kwargs):
        """
        Returns value of factor of the given assignment.

        Example:
        >>> c.factor(P1=True, P2=False)
        0.5
        """

        # Sanity check - our assignment must contain all the nodes in the clique
        if any([key not in kwargs for key in self.nodes]):
            raise ValueError("You need to specify all the values of random"
                             "variables (given {}, needs {}).".format(kwargs, self.nodes))

        key = tuple(kwargs[node] for node in self.nodes)
        return self.potential[key]


class MarkovNetwork(object):

    def __init__(self, cliques):
        """
        A collection of cliques which form a partition of a graph of the Markov
        Network, with their respective factor functions.
        """

        self.cliques = cliques
        self.nodes = reduce(operator.or_, [set(c.nodes) for c in cliques], set())

    def cliques_containing_node(self, node):
        """
        Returns all the cliques that contain given node X_i.
        """

        return [c for c in self.cliques if node in c.nodes]

    def node_probability(self, assignment, node):
        """
        Returns the probability of random variable 'node' being True given
        assignment of the rest of the random variables in the network.
        """

        # We need a new assignment dict so that we don't modify what we were
        # given
        assignment_temp = assignment.copy()
        assignment_temp[node] = True

        product = lambda l: reduce(operator.mul, l, 1)

        nominator = product([c.factor(**assignment_temp)
                             for c in self.cliques_containing_node(node)])
        denominator = 0

        for node_assignment in (True, False):
            assignment_temp[node] = node_assignment
            denominator += product([c.factor(**assignment_temp)
                                    for c in self.cliques_containing_node(node)])

        return float(nominator)/denominator


