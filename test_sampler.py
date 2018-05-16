"""
Contains D-Wave specific tests.
"""

import pytest

from data import IsingModel
from dwave import DWaveSampler
from gibbs import GibbsSampler
from config import DWAVE_SOLVER


class TestDwaveConnection(object):

    def test_connection(self):
        """
        Test that we are able to connect to the D-Wave sampler at all.
        """

        sampler = DWaveSampler()
        obtained = sampler.connection.get_solver(DWAVE_SOLVER)
        assert obtained is not None

    def test_unbiased_checkerboard(self):
        """
        Test that checkerboard problem returns both valid solutions.
        """

        sampler = DWaveSampler()
        checkerboard = IsingModel(J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, h={})
        result = sampler.sample(checkerboard, 10000)

        solutions = [s.as_tuple for s in result]
        assert (-1, 1, -1, 1) in solutions
        assert (1, -1, 1, -1) in solutions

    def test_biased_checkerboard(self):
        """
        Test that biased checkerboard (at one qubit) returns only one solution.
        """

        sampler = DWaveSampler()
        biased_checkerboard = IsingModel(J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, h={0: 2})
        result = sampler.sample(biased_checkerboard, 10000)

        solutions = [s.as_tuple for s in result]
        assert (-1, 1, -1, 1) in solutions
        assert len(solutions) == 1

        # Enforce the opposite checkerboard
        biased_checkerboard = IsingModel(J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, h={0: -2})
        result = sampler.sample(biased_checkerboard, 10000)

        solutions = [s.as_tuple for s in result]
        assert (1, -1, 1, -1) in solutions
        assert len(solutions) == 1


class TestGibbsSampler(object):

    def test_unbiased_checkerboard(self):
        """
        Test that checkerboard problem returns both valid solutions.
        """

        sampler = GibbsSampler()
        checkerboard = IsingModel(J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, h={})
        result = sampler.sample(checkerboard, 10000)

        solutions = [s.as_tuple for s in result]
        assert (-1, 1, -1, 1) in solutions
        assert (1, -1, 1, -1) in solutions

    def test_biased_checkerboard(self):
        """
        Test that biased checkerboard (at one qubit) returns only one solution.
        """

        sampler = DWaveSampler()
        biased_checkerboard = IsingModel(J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, h={0: 2})
        result = sampler.sample(biased_checkerboard, 10000)

        solutions = [s.as_tuple for s in result]
        assert (-1, 1, -1, 1) in solutions
        assert len(solutions) == 1

        # Enforce the opposite checkerboard
        biased_checkerboard = IsingModel(J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, h={0: -2})
        result = sampler.sample(biased_checkerboard, 10000)

        solutions = [s.as_tuple for s in result]
        assert (1, -1, 1, -1) in solutions
        assert len(solutions) == 1
