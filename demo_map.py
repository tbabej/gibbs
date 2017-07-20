"""
Shows capabilities of the Gibbs sampler.
"""

import pprint
import time

from markov import Clique, MarkovNetwork


def main():
    # See illustration on slide 17
    cliques = [
        Clique(['P1'], {
            (True,): 0.2,
            (False,): 1,
        }),
        Clique(['P2'], {
            (True,): 0.2,
            (False,): 1,
        }),
        Clique(['P3'], {
            (True,): 0.2,
            (False,): 1,
        }),
        Clique(['P4'], {
            (True,): 0.2,
            (False,): 1,
        }),
        Clique(['P1', 'P2'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        }),
        Clique(['P2', 'P3'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        }),
        Clique(['P3', 'P4'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        }),
        Clique(['P4', 'P1'], {
            (True, True): 1,
            (True, False): 0.5,
            (False, True): 0.5,
            (False, False): 2,
        })
    ]

    network = MarkovNetwork(cliques)
    print("No patients sick with tuberculosis:")
    print(network.map_query({}))
    print("P1 sick with tuberculosis")
    print(network.map_query({'P1': True}))
    print("P1, P2 sick with tuberculosis")
    print(network.map_query({'P1': True, 'P2': True}))
    print("P1, P2, P3 sick with tuberculosis")
    print(network.map_query({'P1': True, 'P2': True, 'P3': True}))

if __name__ == '__main__':
    main()
