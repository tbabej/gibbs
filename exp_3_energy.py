import seaborn
import pandas
import datetime
import matplotlib.pyplot as plt

from builders import RandomBuilder
from dwave import DWaveSampler
from gibbs import GibbsSampler
from bruteforce import BruteforceSampler

from util import save_experiment


def main():
    res = []

    for width in range(1,3):
        for height in range(1, 2):
            model_width = width * 2
            model_height = height * 2
            decimals = 2
            temperature = 1
            runs = 2

            print("{} x {}".format(model_width, model_height))

            # Create the model
            builder = RandomBuilder(model_width, model_height)
            model = builder.generate(decimals=decimals)

            # First produce a graph with D-Wave
            sampler = DWaveSampler()
            embedding = sampler.find_best_embedding(model.J).data
            results_dwave_prob = [sampler.sample(model, 10000, temperature, embedding=embedding) for _ in range(runs)]

            embedding = builder.embedding_two()
            results_dwave_two = [sampler.sample(model, 10000, temperature, embedding=embedding) for _ in range(runs)]

            embedding = builder.embedding_four()
            results_dwave_four = [sampler.sample(model, 10000, temperature, embedding=embedding) for _ in range(runs)]

            compute_mean = lambda r_list: [r.mean_energy for r in r_list]

            mean_dwave_prob = compute_mean(results_dwave_prob)
            mean_dwave_two = compute_mean(results_dwave_two)
            mean_dwave_four = compute_mean(results_dwave_four)

            print("mean: {}".format(mean_dwave_prob))
            print("mean: {}".format(mean_dwave_two))
            print("mean: {}".format(mean_dwave_four))

            res += [{'type': 'dwaveP','width': model_width, 'height': model_height, 'mean': mean} for mean in mean_dwave_prob]
            res += [{'type': 'dwave2','width': model_width, 'height': model_height, 'mean': mean} for mean in mean_dwave_two]
            res += [{'type': 'dwave4','width': model_width, 'height': model_height, 'mean': mean} for mean in mean_dwave_four]

    save_experiment("mean", {
        'result': res,
    })
    df = pandas.DataFrame.from_dict(res)
    df.to_csv("data_mean_{}.json".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))


if __name__ == '__main__':
    main()
