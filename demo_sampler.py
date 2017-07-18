"""
Shows capabilities of the Gibbs sampler.
"""

def main():
    # See illustration on slide 17
    cliques = [
        Clique(['P1'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P2'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P3'], {
            (True,): 0.2,
            (False,): 100,
        }),
        Clique(['P4'], {
            (True,): 0.2,
            (False,): 100,
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
    sampler = GibbsSampler(network, evidence={'P1':True})

    for sample in sampler:
        # Slow down to peruse the output
        time.sleep(0.1)
        pprint.pprint(sample)


if __name__ == '__main__':
    main()
