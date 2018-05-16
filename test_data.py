import pytest

from data import IsingSample, IsingModel, SamplePool


class TestIsingModel(object):
    """
    Tests the IsingModel class.
    """

    def test_initialization(self):
        """
        Tests that IsingModel can be created.
        """

        h = {0: 20.4, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        # Something is very broken if even this test does not work
        model = IsingModel(J, h)

        assert model is not None

    def test_variables_space_correct(self):
        """
        Tests that IsingModel correctly determines the k-local and 2-local
        space of variables correctly.
        """

        h = {0: 20.4, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)
        assert model.variables == set([0, 1, 2])

        # Present in h, but not present in J
        h = {0: 20.4, 1: 10, 2: -5, 3: 13}
        J = {(0, 1): -4.5, (1, 3): 5}

        model = IsingModel(J, h)
        assert model.variables == set([0, 1, 2, 3])

        # Present in J, but not present in h
        h = {0: 20.4, 1: 10}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)
        assert model.variables == set([0, 1, 2])

    def test_dwave_conversion_h_inter(self):
        """
        Test that the conversion to the D-Wave fills in missing elements in the
        h dictionary.
        """

        h = {0: 20.4, 1: 10, 3: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        h_dwave, J_dwave = IsingModel(J, h).as_dwave()

        assert h_dwave == [20.4, 10, 0, -5]
        assert J_dwave == {(0, 1): -4.5, (1, 2): 5}

    def test_dwave_conversion_big_J_element(self):
        """
        Test that the conversion to the D-Wave fills elements in the h vector
        all the way up to maximum J variable.
        """

        h = {0: 20.4, 1: 10, 4: -5}
        J = {(0, 1): -4.5, (1, 2): 5, (0, 3): 2}

        h_dwave, J_dwave = IsingModel(J, h).as_dwave()

        assert h_dwave == [20.4, 10, 0, 0, -5]
        assert J_dwave == {(0, 1): -4.5, (1, 2): 5, (0, 3): 2}

    def test_serialize_deserialize(self):
        """
        Test that deserialized version of a serialized IsingModel defines the
        same model.
        """

        h = {0: 20.4, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 3): 5}

        serialized = IsingModel(J, h).serialize()
        deserialized = IsingModel.deserialize(serialized)

        assert deserialized.h == h
        assert deserialized.J == J


class TestIsingSample(object):
    """
    Test the IsingSample class.
    """

    def test_valid_sample(self):
        """
        Create valid samples for a given model.
        """

        h = {0: 20.4, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}
        model = IsingModel(J, h)

        # Something is very broken if even this test does not work
        sample = IsingSample(model, {0: 1, 1: -1, 2: 1})
        sample = IsingSample(model, {0: 1, 1: 1, 2: 1}, occurences=54)
        sample = IsingSample(model, {0: -1, 1: 1, 2: -1}, occurences=23)

        assert sample is not None

    def test_invalid_sample_length(self):
        """
        Attempt to create a sample with invalid number of variables.
        """

        h = {0: 20.4, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}
        model = IsingModel(J, h)

        with pytest.raises(ValueError):
            sample = IsingSample(model, {0: -1, 1: 1})
        with pytest.raises(ValueError):
            sample = IsingSample(model, {0: 1, 1: 1, 2: 1, 3: -1})
        with pytest.raises(ValueError):
            sample = IsingSample(model, {})

    def test_sample_energy(self):
        """
        Test that the energy value computation is correct.
        """

        h = {}
        J = {(0, 1): -2.5, (1, 2): 3}
        model = IsingModel(J, h)

        assert IsingSample(model, {0: -1, 1: -1, 2: 1}).energy == pytest.approx(-5.5)

        h = {0: 5.2, 1: 4, 2: -8}
        J = {}
        model = IsingModel(J, h)

        assert IsingSample(model, {0: -1, 1: 1, 2: -1}).energy == pytest.approx(6.8)

        h = {0: 5.2, 1: 4, 2: -8}
        J = {(0, 1): -2.5, (1, 2): 3}
        model = IsingModel(J, h)
        assert IsingSample(model, {0: -1, 1: -1, 2: -1}).energy == pytest.approx(-0.7)

    def test_sample_equality(self):
        """
        Test that the samples are equal iff assignments are equal.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)

        # Equal assignments imply equal samples
        assert IsingSample(model, (-1, 1, 1)) == IsingSample(model, (-1, 1, 1))

        # Unequal assignments imply unequal samples
        assert IsingSample(model, (1, 1, 1))  != IsingSample(model, (1, -1, 1))
        assert IsingSample(model, (1, -1, 1)) != IsingSample(model, (1, 1, -1))
        assert IsingSample(model, (-1, 1, 1)) != IsingSample(model, (1, 1, 1))

    def test_sample_ordering(self):
        """
        Tests that IsingSamples are ordered according to their negative energy.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)

        assert IsingSample(model, (1, 1, 1))  < IsingSample(model, (1, -1, 1))
        assert IsingSample(model, (1, 1, -1)) < IsingSample(model, (-1, -1, 1))
        assert IsingSample(model, (-1, 1, 1)) < IsingSample(model, (1, 1, 1))
        assert IsingSample(model, (1, 1, 1)) < IsingSample(model, (-1, -1, 1))

    def test_sample_deserialize_serialize(self):
        """
        Tests that a sample can be deserialized and then serialized into a
        duplicate object containing same content.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)

        sample = IsingSample(model, (-1, 1, -1))
        deserialized = IsingSample.deserialize(model, sample.serialize())

        # Assert equality
        assert sample == deserialized
        assert sample.energy == deserialized.energy
        assert sample.occurences == deserialized.occurences


class TestVariableClamping(object):
    """
    Test that variable clamping works both in IsingModel and IsingSample.
    """

    def test_clamp_variable_simple(self):
        """
        Makes sure that variable clamping transforms IsingModel in an expected manner.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)
        model.clamp(0, 1)

        assert model.h_clamped == {1: 5.5, 2: -5}
        assert model.J_clamped == {(1, 2): 5}

        model = IsingModel(J, h)
        model.clamp(0, -1)

        assert model.h_clamped == {1: 14.5, 2: -5}
        assert model.J_clamped == {(1, 2): 5}

    def test_clamp_variable_not_present(self):
        """
        Make sure invalid variable cannot be clamped.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)

        with pytest.raises(ValueError):
            model.clamp(8, 1)

    def test_clamp_variable_already_clamped(self):
        """
        Make sure invalid variable cannot be clamped.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)

        # Clamp the variable
        model.clamp(2, -1)

        # Attempt to clamp the variable again
        with pytest.raises(ValueError):
            model.clamp(2, 1)

    def test_clamp_variable_invalid_value(self):
        """
        Make sure invalid variable cannot be clamped.
        """

        h = {0: 4.5, 1: 10, 2: -5}
        J = {(0, 1): -4.5, (1, 2): 5}

        model = IsingModel(J, h)

        # Attempt to clamp with some retarded value
        with pytest.raises(ValueError):
            model.clamp(2, 5)

        with pytest.raises(ValueError):
            model.clamp(2, -3)

        with pytest.raises(ValueError):
            model.clamp(2, 1.01)

        with pytest.raises(ValueError):
            model.clamp(2, -0.99)

        with pytest.raises(ValueError):
            model.clamp(2, 0)

        with pytest.raises(ValueError):
            model.clamp(2, -0.5)

    def test_clamp_multiple_variables(self):
        """
        Make sure clamping multiple variables work.
        """

        h = {0: 4, 1: 10, 2: -5, 3: 4, 4: -5}
        J = {(0, 1): -6, (1, 2): 5, (0, 3): 3, (0, 2): -4}

        model = IsingModel(J, h)

        model.clamp(0, -1)
        model.clamp(2, 1)
        model.clamp(4, -1)

        assert model.h_clamped == {1: 21, 3: 1}
        assert model.J_clamped == {}

    def test_clamp_not_affecting_original(self):
        """
        Make sure that the original ising model definition is not affected by
        clamping variables.
        """

        h = {0: 4, 1: 10, 2: -5, 3: 4, 4: -5}
        J = {(0, 1): -6, (1, 2): 5, (0, 3): 3, (0, 2): -4}

        model = IsingModel(J, h)

        model.clamp(0, -1)
        model.clamp(2, 1)
        model.clamp(4, -1)

        assert model.h == {0: 4, 1: 10, 2: -5, 3: 4, 4: -5}
        assert model.J == {(0, 1): -6, (1, 2): 5, (0, 3): 3, (0, 2): -4}

    def test_clamped_model_sample(self):
        """
        Test that clamped model sample gets the correct energy.
        """

        checkerboard = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={}
        )
        checkerboard.clamp(0, 1)

        sample = IsingSample(checkerboard, (-1, 1, -1))

        assert sample.assignment.as_tuple == (1, -1, 1, -1)
        assert sample.assignment == {0: 1, 1: -1, 2: 1, 3: -1}
        assert sample.energy == -4

        checkerboard = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={0: 5}
        )
        checkerboard.clamp(0, 1)

        sample = IsingSample(checkerboard, (-1, 1, -1))

        assert sample.assignment.as_tuple == (1, -1, 1, -1)
        assert sample.assignment == {0: 1, 1: -1, 2: 1, 3: -1}
        assert sample.energy == 1

        checkerboard = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={0: -3, 3: 2}
        )
        checkerboard.clamp(3, -1)

        sample = IsingSample(checkerboard, (-1, -1, -1))

        assert sample.assignment.as_tuple == (-1, -1, -1, -1)
        assert sample.assignment == {0: -1, 1: -1, 2: -1, 3: -1}
        assert sample.energy == 5

    def test_clamped_model_sample_multple(self):
        """
        Test that clamped model sample with multiple clamped variables gets the
        correct energy.
        """

        checkerboard = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={}
        )
        checkerboard.clamp(1, 1)
        checkerboard.clamp(3, -1)

        sample = IsingSample(checkerboard, (-1, 1))

        assert sample.assignment.as_tuple == (-1, 1, 1, -1)
        assert sample.energy == 0

        checkerboard = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={1: 4}
        )
        checkerboard.clamp(0, -1)
        checkerboard.clamp(1, 1)
        checkerboard.clamp(2, 1)
        checkerboard.clamp(3, -1)

        sample = IsingSample(checkerboard, tuple())

        assert sample.assignment.as_tuple == (-1, 1, 1, -1)
        assert sample.energy == 4


class TestSamplePool(object):
    """
    Tests that the SamplePool works.
    """

    def test_initialize(self):
        """
        Simple sanity test for initialization.
        """

        pool = SamplePool()

        assert len(pool) == 0
        assert pool.n_best(5) == []

    def test_initialize_with_samples(self):
        """
        Initialize SamplePool with a list of existing samples.
        """

        simple = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={0: -8}
        )

        samples = [
            IsingSample(simple, [-1, 1, -1, 1]),
            IsingSample(simple, [1, -1, 1, -1]),
            IsingSample(simple, [-1, -1, -1, -1]),
        ]

        pool = SamplePool(samples)

        assert len(pool) == 3
        assert pool[0] == samples[1]

        # Make sure the entire order is correct
        assert pool[:1000] == sorted(samples, reverse=True)

    def test_initialize_by_adding(self):
        """
        Initialize SamplePool with a list of existing samples.
        """

        simple = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={0: -3}
        )

        samples = [
            IsingSample(simple, [-1, 1, 1, 1]),
            IsingSample(simple, [1, -1, 1, -1]),
            IsingSample(simple, [1, 1, -1, 1]),
            IsingSample(simple, [-1, -1, -1, -1])
        ]

        pool = SamplePool()

        for sample in samples:
            pool.add(sample)

        assert len(pool) == 4
        assert pool[0] == samples[1]

        # Make sure the entire order is correct
        assert pool[:1000] == sorted(samples, reverse=True)

    def test_occurrence_growing(self):
        """
        Check that the number of occurrences is growing when same sample is
        submitted.
        """

        simple = IsingModel(
            J={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1},
            h={0: -3}
        )

        pool = SamplePool()
        pool.add(IsingSample(simple, [1, 1, 1, 1], occurences=3))

        assert len(pool) == 1
        assert pool[0].occurences == 3

        pool.add(IsingSample(simple, [1, 1, 1, 1], occurences=2))

        assert len(pool) == 1
        assert pool[0].occurences == 5

        pool.add(IsingSample(simple, [1, 1, 1, 1]))

        assert len(pool) == 1
        assert pool[0].occurences == 6

        pool.add(IsingSample(simple, [-1, -1, -1, -1], occurences=15))

        assert len(pool) == 2
        assert pool[0].occurences == 6
        assert pool[1].occurences == 15

        pool.add(IsingSample(simple, [-1, -1, -1, -1], occurences=5))

        assert len(pool) == 2
        assert pool[0].occurences == 6
        assert pool[1].occurences == 20
