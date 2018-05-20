import seaborn
import matplotlib.pyplot as plt

from builders import RandomBuilder
from dwave import DWaveSampler
from gibbs import GibbsSampler
from bruteforce import BruteforceSampler


def sample_and_histogram(sampler, model, temperature):
    results = sampler.sample(model, 1000, temperature)
    seaborn.distplot(results.raw_data, rug=False, kde=False, bins=30)
    plt.show()

def main():
    model_width = 4
    model_height = 4

    builder = RandomBuilder(model_width, model_height)
    model = builder.generate(decimals=2)

    # First produce a graph with D-Wave
    sampler = DWaveSampler()
    sample_and_histogram(sampler, model, temperature=1)

    # Then with gibbss sampleler
    sampler = GibbsSampler(n_variables=model_height * model_width)
    sample_and_histogram(sampler, model, temperature=1)

    # Compute partition function of this model
    bruteforcer = BruteforceSampler()
    Z = bruteforcer.partition_function(model)
    print("Partition function: {}".format(Z))

if __name__ == '__main__':
    main()
