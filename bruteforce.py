import itertools
import math

from joblib import Parallel, delayed

from sampler import IsingSampler
from data import IsingSample, SamplePool
from util import split_iterator


class BruteforceSampler(IsingSampler):
    """
    Sample a problem in the ising form using brute force approach.
    """

    def generate_assignments(self, model):
        """
        Generate all possible assignments to the given model.
        """

        # Bruteforce only over information qbits, substitution constraint qubits
        # are generated from the solution so that the constraints hold
        # Using this approach we avoid bruteforcing over invalid solutions.

        info_qbits_num = len(model.variables)
        for assignment in itertools.product([-1, 1], repeat=info_qbits_num):
            yield assignment

    def sample(self, model, num_samples=None, temperature=None):
        """
        Solve the given IsingProblem instance.
        """

        # Create a pool of best solutions and fill it in
        pool = SamplePool()

        with Parallel(n_jobs=-1, verbose=10) as parallel:
            all_assignments = self.generate_assignments(model)
            for chunk_assignments in split_iterator(100000, all_assignments):
                chunk_solutions = parallel(
                    delayed(IsingSample)(model, assignment)
                    for assignment in chunk_assignments
                )

                for solution in chunk_solutions:
                    pool.add(solution)

        return pool

    def partition_function(self, model, temperature):
        """
        Computes the partition function of the given mode.
        """

        return sum([math.e ** (-sample.energy/float(temperature)) for sample in self.sample(model)])
