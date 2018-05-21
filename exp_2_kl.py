import seaborn
import matplotlib.pyplot as plt

from builders import RandomBuilder
from dwave import DWaveSampler
from gibbs import GibbsSampler
from bruteforce import BruteforceSampler

from util import save_experiment


def main():
    #for width in range(3,6):
    #    for height in range(width, 6):
    #        model_width = width
    #        model_height = height
    model_width = 4
    model_height = 4
    decimals = 2
    temperature = 1

    # Create the model
    builder = RandomBuilder(model_width, model_height)
    model = builder.generate(decimals=decimals)

    # Compute partition function of this model
    bruteforcer = BruteforceSampler()
    Z = bruteforcer.partition_function(model, temperature)
    print("Partition function: {}".format(Z))

    # First produce a graph with D-Wave
    sampler = DWaveSampler()
    results = sampler.sample(model, 1000, temperature)

    kl = results.KL_divergence(Z, temperature)
    print("KL divergence: {}".format(kl))

    save_experiment("kl", {
        'width': model_width,
        'height': model_height,
        'model_decimals': decimals,
        'temperature': temperature,
        'dwave_samples': results.raw_data,
        'z': Z,
        'kl': kl
    })


if __name__ == '__main__':
    main()
