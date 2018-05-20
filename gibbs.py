#!/usr/bin/python3
import itertools
import math
import random

import tqdm

from data import IsingSample, SamplePool
from sampler import IsingSampler


class GibbsSampler(IsingSampler):

    """
    A class that performs sampling from multinomial joint distribution given
    all the conditional probability distributions.

    Takes:
      - burnin: A number of steps to throw away at the start of the sampling
                before returning the results.
      - step: Only every step-th sample will be returned.
    """

    def __init__(self, burnin=None, step=None, n_variables=None):
        self.burnin = burnin or int(0.7 * n_variables ** 2)
        self.step = step or int(0.1 * n_variables) + 3

    def node_probability(self, model, assignment, variable, temperature, value=-1):
        """
        Computes the probability of the variable obtaining the value given the
        rest of the assignment.
        """

        possible_samples = []
        for v in (-1, 1):
            z_assignment = assignment.copy()
            z_assignment[variable] = v
            possible_samples.append(IsingSample(model, z_assignment))

        i_assignment = assignment.copy()
        i_assignment[variable] = value
        i_sample = IsingSample(model, i_assignment)

        nominator = math.exp(-1 * i_sample.energy / temperature)
        denominator = sum([
            math.exp(-1 * sample.energy / temperature)
            for sample in possible_samples
        ])

        return nominator / denominator

    def sample(self, model, num_samples, temperature=1):
        """
        Updates each variable using its connditional probability distribution,
        and yields the result.
        """

        # Keep track of the iteration number for thinning and burn-in
        iteration = 0

        # Collect samples
        pool = SamplePool()

        # Create initial assignment
        assignment = {
            key: -1 if random.random() < 0.5 else 1
            for key in model.variables
        }

        progress = tqdm.tqdm(total=self.burnin + num_samples * self.step)
        while len(pool) < num_samples:
            iteration += 1

            for variable in model.variables:
                probability = self.node_probability(model, assignment, variable, temperature)
                sampled_value = -1 if random.random() <= probability else 1
                assignment[variable] = sampled_value

            # Skip first 1000 iterations for burn-in, return every 100th iteration
            # since subsequent samples are correlated
            if iteration >= self.burnin and iteration % self.step == 0:
                sample = IsingSample(model, assignment.copy())
                pool.add(sample)

            progress.update(1)

        return pool
