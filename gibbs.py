#!/usr/bin/python
import operator
import random

import time
import pprint


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


class GibbsSampler(object):

    """
    A class that performs sampling from multinomial joint distribution given
    all the conditional probability distributions.

    Takes:
      - network: An instance of MarkovNetwork
      - evidence: A dictionary containing observed data
      - burnin: A number of steps to throw away at the start of the sampling
                before returning the results.
      - step: Only every step-th sample will be returned.

    Example:
    >>> network = MarkovNetwork([
        Clique(['P1', 'P2'], {
               (True, True): 1,
               (True, False): 0.5,
               (False, True): 0.5,
               (False, False): 2,
        }),
        ...
    ])
    >>> evidence = {'P1': True}
    >>> g = GibbsSampler(network, evidence)
    """

    def __init__(self, network, evidence, burnin=1000, step=100):
        self.network = network
        self.evidence = evidence
        self.burnin = burnin
        self.step = step

        # Choose uniform assigmnemt for all variables and override
        # with observed data
        self.assignment = {
            key: True if random.random() < 0.5 else False
            for key in self.network.nodes
        }
        self.assignment.update(evidence)

        # During Gibbs sampling, we only update unobserved variables
        self.unobserved_vars = set(self.network.nodes) - set(evidence.keys())

    def __iter__(self):
        """
        Updates each variable using its connditional probability distribution,
        and yields the result.
        """

        # Gibbs sampling runs indefinitely
        iteration = 0

        while True:
            iteration += 1

            for variable in self.unobserved_vars:
                probability = self.network.node_probability(self.assignment,
                                                            variable)
                sampled_value = True if random.random() <= probability else False
                self.assignment[variable] = sampled_value

            # Skip first 1000 iterations for burn-in, return every 100th iteration
            # since subsequent samples are correlated
            if iteration >= self.burnin and iteration % self.step == 0:
                yield self.assignment.copy()

    def generate(self, num_samples):
        """
        Returns a list of samples.
        """

        return list(islice(self, num_samples))


def main():
    # See illustration on slide 17
    cliques = [
        Clique(['P1'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P2'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P3'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P4'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P1', 'P2'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        }),
        Clique(['P2', 'P3'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        }),
        Clique(['P3', 'P4'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        }),
        Clique(['P4', 'P1'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        })
    ]

    network = MarkovNetwork(cliques)
    sampler = GibbsSampler(network, evidence={'P1':True})

    for sample in sampler:
        # Slow down to peruse the output
        time.sleep(1)
        pprint.pprint(sample)


if __name__ == '__main__':
    main()
