import seaborn
import matplotlib.pyplot as plt

from builders import RandomBuilder
from dwave import DWaveSampler
from gibbs import GibbsSampler
from bruteforce import BruteforceSampler
from util import save_experiment


def sample_and_histogram(sampler, model, temperature):
    results = sampler.sample(model, 1000, temperature)
    seaborn.distplot(results.raw_data, rug=False, kde=False, bins=30)
    plt.show()
    return results

def main():
    model_width = 4
    model_height = 4
    temperature = 1
    decimals = 2

    builder = RandomBuilder(model_width, model_height)
    model = builder.generate(decimals=decimals)

    # First produce a graph with D-Wave
    sampler = DWaveSampler()
    results_dwave = sample_and_histogram(sampler, model, temperature)

    # Then with gibbss sampleler
    sampler = GibbsSampler(n_variables=model_height * model_width)
    results_gibbs = sample_and_histogram(sampler, model, temperature)

    save_experiment("histogram", {
        'width': model_width,
        'height': model_height,
        'model_decimals': decimals,
        'temperature': temperature,
        'dwave_samples': results_dwave.raw_data,
        'gibbs_samples': results_gibbs.raw_data
    })

if __name__ == '__main__':
    main()
