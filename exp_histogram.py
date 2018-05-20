import seaborn
import matplotlib.pyplot as plt

from builders import RandomBuilder
from dwave import DWaveSampler
from gibbs import GibbsSampler
from bruteforce import BruteforceSampler


def sample_and_histogram(sampler, model, temperature):
    # First D-wave
    results = sampler.sample(model, 1000, temperature)
    histogram = results.to_energy_histogram()

    # Seaborn expects non-histogramed data for visualization
    print(sum(histogram.values()))
    raw_data = sum([[key] * value for key, value in histogram.items()], [])
    print(len(raw_data))

    seaborn.distplot(raw_data, rug=False, kde=False, bins=30)
    plt.show()


def determine_gibbs_params(n_variables):
    burn_in = int(0.7 * n_variables ** 2)
    thinning = int(0.1 * n_variables) + 3

    return burn_in, thinning


def main():
    model_width = 4
    model_height = 4

    builder = RandomBuilder(model_width, model_height)
    model = builder.generate(decimals=2)

    # First produce a graph with D-Wave
    sampler = DWaveSampler()
    sample_and_histogram(sampler, model, temperature=1)

    # Then with gibbss sampleler
    gibbs_params = determine_gibbs_params(model_width * model_height)
    sampler = GibbsSampler(*gibbs_params)
    sample_and_histogram(sampler, model, temperature=1)

    # Compute partition function of this model
    bruteforcer = BruteforceSampler()
    Z = bruteforcer.partition_function(model)
    print("Partition function: {}".format(Z))

if __name__ == '__main__':
    main()
