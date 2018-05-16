import random
import networkx
import logger
import numpy

from PIL import Image
import matplotlib.pyplot as plt

from data import IsingModel


class GridBuilder(logger.LoggerMixin):
    """
    Base class for nearest-neighbour pairwise Markov Network models.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.graph = networkx.grid_2d_graph(width, height)

    def variable(self, x, y):
        return y * self.width + x

    def coordinates(self, variable):
        return (variable % self.width, variable // self.width)

    def generate(self):
        raise NotImplemented

    def visualize(self, sample):
        matrix = numpy.zeros((self.width, self.height))
        for key, value in sample.assignment.items():
            matrix[tuple(reversed(self.coordinates(key)))] = value

        plt.matshow(matrix)
        plt.show()


class RandomBuilder(GridBuilder):
    """
    Constructs an Nearest-neighbour pairwise grid Markov Network with random
    biases and couplings.
    """

    def generate(self):
        J = {}
        h = {}

        for (node1, node2) in self.graph.edges:
            J[self.variable(*node1), self.variable(*node2)] = 2 * random.random() - 1
        for node in self.graph.nodes:
            h[self.variable(*node)] = 2 * random.random() - 1

        return IsingModel(J, h)


class ImageBuilder(GridBuilder):
    """
    Construct a nearest-neighbour pairwise grid Markov Network based on a image
    template.
    """

    def __init__(self, filename):
        self.image = Image.open(filename)
        self.width = self.image.width
        self.height = self.image.height

        self.graph = networkx.grid_2d_graph(self.width, self.height)

    def binpix(self, x, y):
        """
        Returns the binarized value of a pixel, -1 stands for black,
        1 for white.
        """

        return 1 if self.image.getpixel((x, y)) > 127 else -1

    def generate(self):
        h = {}
        J = {}

        for (node1, node2) in self.graph.edges:
            value = (-1) * self.binpix(*node1) * self.binpix(*node2)
            J[self.variable(*node1), self.variable(*node2)] = value

        return IsingModel(J, h)
