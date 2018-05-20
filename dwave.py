from collections import defaultdict

import joblib
import numpy as np

from dwave_sapi2.core import solve_ising
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer
from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.util import get_hardware_adjacency, get_chimera_adjacency

import config
from sampler import IsingSampler
from data import IsingSample, SamplePool


class Embedding(object):
    """
    Represents an embedding of a model into a given graph.
    """

    def __init__(self, embedding_data):
        """
        Takes embedding defined as a list of lists, and compute statistics over it.
        """

        self.data = embedding_data
        self.qubits_used = set([node for sublist in embedding_data for node in sublist])
        self.chain_lengths = [len(sublist) for sublist in embedding_data]

        # Derive statistics
        self.no_qubits = len(self.qubits_used)
        self.max_chain_length = max(self.chain_lengths)
        self.avg_chain_length = np.average(self.chain_lengths)


class DWaveSampler(IsingSampler):
    """
    Samples a PGM using D-Wave's quantum annealer.
    """

    def __init__(self):
        self.connection = RemoteConnection(
            config.DWAVE_SAPI_URL,
            config.DWAVE_TOKEN,
            config.DWAVE_PROXY
        )
        self.solver = self.connection.get_solver(config.DWAVE_SOLVER)
        self.adjacency_matrix = get_hardware_adjacency(self.solver)

    def find_best_embedding(self, J, improvements=100, runs=4):
        """
        Since find_embedding is randomized, attempt to find embedding several
        times and pick the best result.
        """

        # Generate multiple embeddings in parallel
        raw_embeddings = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(find_embedding)
                (J.keys(), self.adjacency_matrix, max_no_improvement=improvements)
            for i in range(runs)
        )
        embeddings = [Embedding(e) for e in raw_embeddings]

        # Pick the best embedding
        best_embedding = None

        for embedding in embeddings:
            embedding_is_superior = best_embedding is None or any([
                best_embedding.max_chain_length > embedding.max_chain_length,
                best_embedding.avg_chain_length > embedding.avg_chain_length
                    and best_embedding.max_chain_length == embedding.max_chain_length,
                best_embedding.no_qubits > embedding.no_qubits
                    and best_embedding.max_chain_length == embedding.max_chain_length
                    and best_embedding.avg_chain_length == embedding.avg_chain_length
            ])

            if embedding_is_superior:
                best_embedding = embedding
                self.info("New embedding: {embedding.no_qubits}, max chain: {embedding.max_chain_length}, avg chain: {embedding.avg_chain_length}".format(embedding=embedding))

        return best_embedding

    def query_dwave(self, h, J, embedding, samples, temperature, batch_size):
        """
        Queries D-Wave multiple times for solution of the given Ising model,
        aggregating the unembedded results.
        """

        results = {
            'energies': [],
            'solutions': [],
            'num_occurrences': [],
            'timing': []
        }

        num_batches = samples // batch_size

        for i in range(num_batches):
            batch_solved = False
            while not batch_solved:
                try:
                    self.info("Sampling batch {i}".format(i=i))
                    batch = solve_ising(
                        self.solver, h, J,
                        answer_mode='histogram',
                        auto_scale=True,
                        num_reads=batch_size,
                        num_spin_reversal_transforms=5,
                        beta=temperature,
                        postprocess='sampling',
                        chains=embedding
                    )
                    batch_solved = True
                    self.info("Done")
                except Exception as e:
                    self.verbose(str(e))
                    self.info("Exception occured, retrying...")

            # Collect the batch results
            batch_solutions = unembed_answer(
                batch['solutions'],
                embedding,
                broken_chains='vote'
            )

            results['solutions'].extend(batch_solutions)
            results['num_occurrences'].extend(batch['num_occurrences'])

        # Aggregate the same unembedded answers
        aggregated = defaultdict(dict)
        data = zip(results['solutions'], results['num_occurrences'])

        for result, count in data:
            key = tuple(result)
            result_data = aggregated.get(key, {})

            # Recompute the average solution energy
            prior_count = result_data.get('count', 0)
            result_data['count'] = prior_count + count

            aggregated[key] = result_data

        # Return as a sorted list
        aggregated_list = [(key, value['count'])for key, value in aggregated.items()]
        return list(sorted(aggregated_list, key=lambda x: x[1]))

    def sample(self, model, num_samples, temperature=1, batch_size=None, embedding=None):
        # Determine the batch size
        batch_size = batch_size or min(10000, num_samples)
        # Extract the model and get h and J formatted for D-Wave API
        h_dwave, J_dwave = model.as_dwave()

        # Find the embedding
        if embedding is None:
            embedding = self.find_best_embedding(J_dwave).data
            #self.info(embedding)

        # Transform J and h using found graph embedding
        # embed_model can still do some changes to the embedding
        h_embedded, J_embedded, J_couplings, final_embedding = embed_problem(
            h_dwave, J_dwave,
            embedding,
            adj=self.adjacency_matrix,
            h_range=(-2, 2),
            j_range=(-1, 1)
        )

        # Compute max coefficient
        max_coefficient = max([
            abs(max(h_embedded)),
            abs(min(h_embedded)),
            abs(max(J_embedded.values())),
            abs(min(J_embedded.values()))
        ])

        # Update J matrix
        J_embedded.update({key: -1.0 * max_coefficient for key in J_couplings.keys()})

        results = self.query_dwave(
            h_embedded,
            J_embedded,
            final_embedding,
            num_samples,
            temperature,
            batch_size
        )

        samples = [
            IsingSample(model, assignment, occurences)
            for (assignment, occurences) in results
        ]

        # Return as a sorted list
        sorted_solutions = SamplePool(samples)

        return sorted_solutions
