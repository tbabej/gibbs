import pandas
import datetime
import seaborn
import matplotlib.pyplot as plt

from builders import RandomBuilder
from dwave import DWaveSampler
from gibbs import GibbsSampler
from bruteforce import BruteforceSampler

from util import save_experiment


def main():
    res = []

    for width in range(1,2):
        for height in range(width, 3):
            model_width = width * 2
            model_height = height * 2
            decimals = 2
            temperature = 1
            runs = 2
            print("{} x {}".format(model_width, model_height))

            # Create the model
            builder = RandomBuilder(model_width, model_height)
            model = builder.generate(decimals=decimals)

            # Compute partition function of this model
            bruteforcer = BruteforceSampler()
            Z = bruteforcer.partition_function(model, temperature)

            # First produce a graph with D-Wave
            sampler = DWaveSampler()
            results_dwave_prob = [sampler.sample(model, 10000, temperature) for _ in range(runs)]

            embedding = builder.embedding_two()
            results_dwave_two = [sampler.sample(model, 10000, temperature, embedding=embedding) for _ in range(runs)]

            embedding = builder.embedding_four()
            results_dwave_four = [sampler.sample(model, 10000, temperature, embedding=embedding) for _ in range(runs)]

            sampler = GibbsSampler(n_variables=model_width*model_height)
            results_gibbs = [sampler.sample(model, 10000, temperature) for _ in range(runs)]

            compute_kl = lambda r_list: [r.KL_divergence(Z, temperature) for r in r_list]

            kl_gibbs = compute_kl(results_gibbs)
            kl_dwave_prob = compute_kl(results_dwave_prob)
            kl_dwave_two = compute_kl(results_dwave_two)
            kl_dwave_four = compute_kl(results_dwave_four)

            print("KL divergence: {}".format(kl_gibbs))
            print("KL divergence: {}".format(kl_dwave_prob))
            print("KL divergence: {}".format(kl_dwave_two))
            print("KL divergence: {}".format(kl_dwave_four))

            res += [{'type': 'gibbs', 'width': model_width, 'height': model_height, 'kl': kl} for kl in kl_gibbs]
            res += [{'type': 'dwaveP','width': model_width, 'height': model_height, 'kl': kl} for kl in kl_dwave_prob]
            res += [{'type': 'dwave2','width': model_width, 'height': model_height, 'kl': kl} for kl in kl_dwave_two]
            res += [{'type': 'dwave4','width': model_width, 'height': model_height, 'kl': kl} for kl in kl_dwave_four]

            df = pandas.DataFrame.from_dict(res)
            df.to_csv("data_kl_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            df.to_json("data_kl_{}.json".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            save_experiment("kl", {
                'result': res,
            })


if __name__ == '__main__':
    main()
