import bisect
import collections
import copy
import decimal

from . import bucket
from . import null


class Histogram():

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.buckets = []
        self.null_frac = decimal.Decimal(0)

    def __copy__(self):
        hist = Histogram(self.m, self.n)
        hist.buckets = [copy.copy(b) for b in self.buckets]
        hist.null_frac = self.null_frac
        return hist

    def __len__(self):
        return len(self.buckets)

    def __getitem__(self, given):
        if isinstance(given, slice):
            sub_hist = Histogram(self.m, self.n)
            sub_hist.buckets = self.buckets[given]
            return sub_hist
        return self.buckets[given]

    def __mul__(self, other):
        """Multiplies two histograms together

        The computation is somewhat heuristic but it checks out.
        """

        hist = copy.copy(self)

        for i, bucket in enumerate(hist.buckets):
            if bucket.cardinality == 1:
                p = other.p(bucket.left)
            else:
                _, buckets = other.find_buckets(bucket.left, bucket.right)
                p = sum(b.frequency for b in buckets) / len(buckets)

            hist.buckets[i].frequency *= p

        # TODO: what should we do with Nones?

        return hist

    def __eq__(self, other):
        if len(self) != len(other) or self.null_frac != other.null_frac:
            return False
        for b1, b2 in zip(self.buckets, other.buckets):
            if b1 != b2:
                return False
        return True

    def fit(self, values):

        if not values:
            raise ValueError('values is an empty sequence')

        # Count the occurences of each value
        counter = collections.Counter(values)

        # Store the number of null values
        try:
            self.null_frac = decimal.Decimal(counter.pop(null.Null()))
        except KeyError:
            pass

        # Store the m most frequent values
        self.buckets = []
        for val, count in counter.most_common(self.m):
            self.buckets.append(bucket.Bucket(val, val, count, 1))
            counter.pop(val)

        # Store the rest of the values in n equi-height buckets
        if self.n > 0:
            height = (len(values) - sum(b.frequency for b in self.buckets) - self.null_frac) / self.n
            buck = None

            for val, count in sorted(counter.items()):

                if buck is None:
                    buck = bucket.Bucket(val, val, count, 1)
                else:
                    buck += bucket.Bucket(val, val, count, 1)

                if buck.frequency >= height:
                    self.buckets.append(buck)
                    buck = None

            if buck is not None:
                self.buckets.append(buck)

        # Convert the counts to probabilities
        total = decimal.Decimal(str(len(values)))
        for i, _ in enumerate(self.buckets):
            self[i].frequency /= total
        self.null_frac /= total

        # Sort the buckets so that binary search can be applied
        self.buckets.sort(key=lambda x: x.left)

        return self

    def find_buckets(self, left, right):
        """Returns the buckets that contain at least one value in [left, right]."""
        indexes = []
        buckets = []
        for i, buck in enumerate(self.buckets):
            if buck.right >= left and buck.left <= right:
                indexes.append(i)
                buckets.append(buck)
        return indexes, buckets

    def find_bucket(self, val):
        """Returns the bucket that contains val using binary search."""
        if val < self[0] or val > self[-1]:
            return -1, None

        i = bisect.bisect_left(self, val)

        if val < self[i] or val > self[i]:
            return -1, None

        return i, self[i]

    def p(self, val):
        """Returns P(val)."""
        if val is None:
            return self.null_frac

        _, buck = self.find_bucket(val)

        if buck is None:
            return 0

        return buck.frequency / buck.cardinality

    def __str__(self):
        return '\n'.join(str(b) for b in self.buckets) + \
            (f'\nNone: {self.null_frac:.5f}' if self.null_frac > 0 else '')

    def __repr__(self):
        return str(self)



def new_histogram(buckets, null_frac=0):
    """Returns a Histogram with the given buckets.

    This should only be used for testing purposes.
    """
    m = sum(1 for b in buckets if b.cardinality == 1)
    n = sum(1 for b in buckets if b.cardinality > 1)
    hist = Histogram(m, n)
    hist.buckets = buckets
    hist.null_frac = decimal.Decimal(null_frac)
    return hist
