import functools
import heapq
import json
import operator
import numpy

from collections import defaultdict
from util import symmetriczerodefaultdict, zerodefaultdict, hashabledict
from logger import LoggerMixin


class IsingModel(LoggerMixin):
    """
    An object representing a given Ising model.
    """

    def __init__(self, J, h):
        """
        Initialize a given Ising model. Takes:
        - J: a dictionary representing non-zero elements of the J matrix
        - h: a dictionary representing non-zero elements of the h vector
        """

        # Stores the original model definition
        self.J = symmetriczerodefaultdict(J)
        self.h = zerodefaultdict(h)

        # Stores adjusted J and h matrix with clamped nodes taken into account
        self.J_clamped = symmetriczerodefaultdict(J)
        self.h_clamped = zerodefaultdict(h)

        # Store values of clamped nodes
        self.clamped = {}
        # Store the energy offset due to clamped qubits so that we correctly
        # compute resulting energy
        self.energy_offset = 0

        self.variables = self.J.elements | set(self.h)

    def clamp(self, variable, value):
        """
        Fix the given variable to the given value. This removes relevant
        entries from the h vector and converts entries in the J matrix to
        biases in the h vector.
        """

        self.debug("Fixing {} to {}".format(variable, value))

        if variable in self.clamped:
            if value != self.clamped[variable]:
                raise ValueError("Cannot clamp {} to {}. Variable is already clamped to a different value: {}".format(variable, value, self.clamped[variable]))
            else:
                # Assume no-op
                return

        if variable not in self.variables:
            raise ValueError("Variable not in available in this model: {}".format(variable))

        if value not in (1, -1):
            raise ValueError("Invalid value: {}".format(value))

        # Drop the variable from the k-local space and raw space
        self.variables.discard(variable)

        # Record the clamped variable
        self.clamped[variable] = value

        # Drop the entry in the h vector
        self.energy_offset += self.h_clamped.pop(variable, 0) * value

        # Generate linear terms
        for edge, coupling in list(self.J_clamped.items()):
            if variable in edge:
                variable2 = edge[1] if edge[0] == variable else edge[0]

                if variable2 not in self.clamped:
                    self.h_clamped[variable2] += coupling * value
                else:
                    self.energy_offset += self.clamped[variable2] * coupling * value

                # Remove the key from the J matrix
                self.J_clamped.pop(edge)

    def copy(self):
        """
        Creates a copy of the given IsingModel.
        """

        copied_model = IsingModel(
            J=self.J.copy(),
            h=self.h.copy()
        )

        copied_model.J_clamped = self.J_clamped.copy()
        copied_model.h_clamped = self.h_clamped.copy()
        copied_model.clamped = self.clamped.copy()
        copied_model.variables = self.variables.copy()
        copied_model.energy_offset = self.energy_offset

        return copied_model

    def as_dwave(self):
        """
        Reformulate the model to D-Wave representation,
        which uses the same J matrix, but uses an space-inefficient h vector
        """

        # Compute the variable range
        mapping = {
            variable: index
            for index, variable in enumerate(sorted(self.variables))
        }

        h_dwave = [0] * len(self.variables)
        J_dwave = {}

        # Shift all values so thae first qubit is 0
        for (node1, node2), value in self.J_clamped.items():
            J_dwave[(mapping[node1], mapping[node2])] = value

        for index, value in self.h_clamped.items():
            h_dwave[mapping[index]] = value

        return h_dwave, J_dwave

    def as_data_table(self):
        """
        Reformulate the model to the text representation.

        Example:
            2048 3
            0 0 1.4
            1 1 -0.5
            2 2 1.3
            1 2 2.5
        """

        lines = []
        for node, value in self.h_clamped.items():
            lines.append("{node} {node} {value:0.4f}"
                         .format(node=node, value=value))

        for term_index, value in self.J_clamped.items():
            lines.append("{term} {value:0.4f}"
                         .format(term=' '.join(term_index), value=value))

        introline = "2048 {length}".format(length=len(lines))

        return '\n'.join([introline] + lines)

    def serialize(self):
        """
        Returns a serialized version of the IsingModel instance.
        """

        # Function to convert a tuple to a list, if it is a tuple
        tuple_to_list = lambda l: list(l) if isinstance(l, tuple) else l

        serialize_items = lambda d: [
            "{}: {}".format(tuple_to_list(key), tuple_to_list(value))
            for key, value in d.items()
        ]

        data = {
            'J': serialize_items(self.J),
            'h': serialize_items(self.h),
            'J_clamped': serialize_items(self.J_clamped),
            'h_clamped': serialize_items(self.h_clamped),
            'variables': list(self.variables),
            'energy_offset': self.energy_offset,
            'clamped': serialize_items(self.clamped),
        }

        return json.dumps(data)

    @classmethod
    def deserialize(cls, datastring):
        """
        Constructs a IsingModel instance given its serialized representation.
        """

        data = json.loads(datastring)

        # Processing a list of ["(1,2): 5.7", "(3,5): 4.2", ...]
        deserialize_tuple_items = lambda l: {
            tuple(json.loads(item.split(':')[0])): json.loads(item.split(':')[1])
            for item in l
        }

        # Processing a list of ["4: 5.7", "9: 4.2", ...]
        deserialize_single_items = lambda l: {
            int(item.split(':')[0]): json.loads(item.split(':')[1])
            for item in l
        }

        # Load basic data
        J = deserialize_tuple_items(data['J'])
        h = deserialize_single_items(data['h'])

        # Load the object
        obj = cls(J, h)

        # Load clamped model versions
        J_clamped = deserialize_tuple_items(data['J_clamped'])
        h_clamped = deserialize_single_items(data['h_clamped'])

        obj.J_clamped = symmetriczerodefaultdict(J_clamped)
        obj.h_clamped = zerodefaultdict(h_clamped)

        # Load additional attributes
        obj.clamped = deserialize_single_items(data['clamped'])
        obj.variables = set(data['variables'])
        obj.energy_offset = data['energy_offset']

        return obj


@functools.total_ordering
class IsingSample(object):
    """
    An object representing a sample to a given Ising model.
    """

    def __init__(self, model, assignment, occurences=1):
        """
        Initialize a given Ising model sample. Takes:
        - model:  An instance of IsingModel class.
        - assignment: A dictionary of variable values. The length of the list must be
                    equal to the number of free variables in the IsingModel.
        - occurences: Number of samples observed with this assignment.
        """

        self.model = model
        self.occurences = occurences

        # If sample is being initialized from a list or a tuple, convert it
        # to a dictionary, assuming all variables from k-local space are covered
        if isinstance(assignment, tuple) or isinstance(assignment, list):
            variable_keys = sorted(model.variables)
            assignment = {
                key: value
                for key, value in zip(variable_keys, assignment)
            }

        # Verify that all variables in the k-local space are covered
        missing_variables = model.variables - set(assignment.keys())
        if missing_variables:
            raise ValueError(
                "Missing variables in the sample: {}".format(
                    ','.join(map(str, missing_variables))
                )
            )

        extra_variables = set(assignment.keys()) - model.variables
        if extra_variables:
            raise ValueError(
                "Sample containes unexpected variables: {}".format(
                    ','.join(map(str, extra_variables))
                )
            )

        # Store the sample
        self.assignment = hashabledict(self.expand_sample(model, assignment))

        # Cache the energy of the sample
        self.energy = self.compute_energy()

    @staticmethod
    def expand_sample(model, sample):
        """
        Compute the expanded version of the sample with the clamped variables.
        """

        # Fill in the clamped variable values
        sample.update(model.clamped)

        return sample

    def compute_energy(self):
        """
        Return the energy of the given sample.
        """

        energy = self.model.energy_offset

        for key, value in self.model.J_clamped.items():
            energy += functools.reduce(operator.mul, [self.assignment[k] for k in key], value)

        for key, value in self.model.h_clamped.items():
            energy += self.assignment[key] * value

        return energy

    @property
    def as_tuple(self):
        """
        Return sample as tuple.
        """
        return self.assignment.as_tuple

    def serialize(self):
        """
        Returns a serialized version of the IsingModel instance.
        """

        return json.dumps((
            list(sorted(self.assignment.items())),
            self.occurences,
            self.energy
        ))

    @classmethod
    def deserialize(cls, model, datastring):
        """
        Constructs a IsingSample instance given its serialized representation.
        """

        data = json.loads(datastring)
        sample_dict = {int(key): value for key, value in data[0]}
        return cls(model, sample_dict, data[1])

    def __gt__(self, other):
        """
        Method to compare different samples of the same IsingModel.
        """

        # For performance reasons, we are not enforcing the fact that the
        # IsingModels are actually the same.

        if self.energy != other.energy:
            return self.energy < other.energy
        else:
            return self.assignment < other.assignment

    def __eq__(self, other):
        """
        Two IsingSamples are considered the same if they encode the same sample.
        """

        return self.assignment == other.assignment

    def __repr__(self):
        """
        Return a string representation of this object.
        """

        template = (
            "Sample {sol}: {occur} occurences, energy: {logical}"
        )

        return template.format(
            sol="[" + ' '.join([" 1," if var == 1 else "-1," for var in self.as_tuple]) + "]",
            occur=self.occurences,
            logical=self.energy,
        )


class SamplePool(object):
    """
    A data structure to keep the N best samples.
    """

    def __init__(self, data=None):
        """
        Initialize the SamplePool.
        """

        # We're using both heap (for sorting and min-max extraction) as well as
        # dictionary (for easy access to the samples in the pool. Memory
        # overhead is small as dictionary only stores pointers to the
        # IsingSample objects.

        self.heap = list()
        self.dict = dict()

        # Rearrange the data so that it is ordered as a heap
        data = data or []
        for sample in data:
            self.add(sample)

    def __len__(self):
        """
        Return the number of samples in the pool.
        """

        return sum([sample.occurences for sample in self.heap])

    def __getitem__(self, key):
        """
        Returns the slices of the pool.
        """

        return self.heap[key]

    def add(self, sample):
        """
        Add the Sample to the pool, making sure to deduplicate by aggregation.
        """

        # First check if the sample is already in the pool, if yes, just
        # add the occurences
        if sample.as_tuple in self.dict:
            present = self.dict[sample.as_tuple]
            present.occurences += int(sample.occurences)
            return

        heapq.heappush(self.heap, sample)
        self.dict[sample.as_tuple] = sample

    def n_best(self, number):
        """
        Retrieve the N best samples.
        """

        return heapq.nlargest(number, self.heap)

    def to_energy_histogram(self):
        """
        Bins the samples according to their energy.
        """

        histogram = defaultdict(int)
        for sample in self.heap:
            histogram[sample.energy] += sample.occurences

        return histogram

    @property
    def raw_data(self):
        """
        Return the raw sample list.
        """

        return sum([
            [sample.energy] * sample.occurences
            for sample in self.heap
        ], [])

    @property
    def mean_energy(self):
        """
        Return the mean energy in the pool.
        """

        return numpy.average(self.raw_data)

    def KL_divergence(self, partition_function, temperature):
        """
        Return the KL divergence to the idealized Boltzmann distribution.
        """

        num_samples = len(self)

        boltz_prob = lambda sample: (math.e ** (-sample.energy/temperature)) / partition_function
        data_prob = lambda sample: sample.occurences / float(num_samples)

        return sum([
            data_prob(sample) * math.log(data_prob(sample) / boltz_prob(sample))
            for sample in self.heap
        ])
