from collections import defaultdict

class hashabledict(dict):
    """
    A special version of a dict which is hashable.
    """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    @property
    def as_tuple(self):
        return tuple(value for index, value in sorted(self.items()))


class zerodefaultdict(defaultdict):
    """
    A special version of a dict, that:
        * Return 0.0 when querying unknown keys
    """

    def __init__(self, other=None):
        defaultdict.__init__(self)
        if other:
            self.update(other)

    @staticmethod
    def __missing__(key):
        return 0.0

    def get(self, key, default=0.0):
        return defaultdict.get(self, key, default)

    def copy(self):
        return zerodefaultdict(self)


class symmetriczerodefaultdict(zerodefaultdict):
    """
    A special version of a dict, that:
        * Return 0.0 when querying unknown keys
        * Expects any tuple keys and treats them as symmetrical
    """

    def __init__(self, other=None):
        zerodefaultdict.__init__(self)
        if other:
            self.update(other)

    def get(self, key, default=0.0):
        key = tuple(sorted(key))
        return zerodefaultdict.get(self, key, default)

    def __getitem__(self, key):
        key = tuple(sorted(key))
        return zerodefaultdict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if key.count(key[0]) == len(key):
            raise ValueError("We can't set diagonal elements. You tried to set {}".format(key))
        key = tuple(sorted(key))
        return zerodefaultdict.__setitem__(self, key, value)

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def copy(self):
        return symmetriczerodefaultdict(self)

    def pop(self, key):
        key = tuple(sorted(key))
        return zerodefaultdict.pop(self, key)

    @property
    def elements(self):
        used_rows = set([index[0] for index in self])
        used_cols = set([index[1] for index in self])

        return used_rows | used_cols
