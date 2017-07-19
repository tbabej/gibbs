#!/usr/bin/python3
import itertools
import random


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

        return list(itertools.islice(self, num_samples))
