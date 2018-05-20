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
        raise NotImplementedError("Generate method needs to be overridden")

    def visualize(self, sample):
        matrix = numpy.zeros((self.width, self.height))
        for key, value in sample.assignment.items():
            matrix[tuple(reversed(self.coordinates(key)))] = value

        plt.matshow(matrix)
        plt.show()

    def embedding_two(self):
        if self.height > 32 or self.width > 32:
            raise ValueError("Graph size is out of bounds")

        if self.height % 2 != 0 or self.width % 2 != 0:
            raise ValueError("Both dimensions need to be even for 2-embedding")

        embedding_map = {}

        block_height = self.height / 2
        block_width = self.width / 2

        for block_y in range(block_height):
            for block_x in range(block_width):
                upper_left = self.variable(2*block_x, 2*block_y)
                upper_right = self.variable(2*block_x + 1, 2*block_y)
                lower_left = self.variable(2*block_x, 2*block_y + 1)
                lower_right = self.variable(2*block_x + 1, 2*block_y + 1)

                block_hardware_offset = 8 * 16 * block_y + 8 * block_x
                embedding_map[upper_left] = [block_hardware_offset, block_hardware_offset + 4]
                embedding_map[upper_right] = [block_hardware_offset + 1, block_hardware_offset + 5]
                embedding_map[lower_right] = [block_hardware_offset + 2, block_hardware_offset + 6]
                embedding_map[lower_left] = [block_hardware_offset + 3, block_hardware_offset + 7]

        return [value for key, value in sorted(embedding_map.items())]

    def embedding_four(self):
        if self.height > 16 or self.width > 32:
            raise ValueError("Graph size is out of bounds")

        if self.width % 2 != 0:
            raise ValueError("Width needs to be even for 4-embedding")

        embedding_map = {}

        block_height = self.height
        block_width = self.width / 2

        for block_y in range(block_height):
            for block_x in range(block_width):
                left = self.variable(2*block_x, block_y)
                right = self.variable(2*block_x + 1, block_y)

                block_hardware_offset = 8 * 16 * block_y + 8 * block_x
                embedding_map[left] = [
                    block_hardware_offset,
                    block_hardware_offset + 3,
                    block_hardware_offset + 4,
                    block_hardware_offset + 7,
                ]
                embedding_map[right] = [
                    block_hardware_offset + 1,
                    block_hardware_offset + 2,
                    block_hardware_offset + 5,
                    block_hardware_offset + 6,
                ]

        return [value for key, value in sorted(embedding_map.items())]


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
