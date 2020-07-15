import collections
import copy
import decimal

from . import histogram


def argsort(arr):
    return [i[0] for i in sorted(enumerate(arr), key=lambda x:x[1])]


class CPD():

    def __init__(self, by_m, by_n, on_m, on_n):
        self.by_m = by_m  # Number of MCVs used for the parent
        self.by_n = by_n  # Number of bins used for the parent
        self.on_m = on_m  # Number of MCVs used for the child
        self.on_n = on_n  # Number of bins used for the child
        self.by_hist = None
        self.on_hists = None
        self.on_null_hist = None

    def __len__(self):
        return len(self.by_hist)

    def __getitem__(self, given):
        if isinstance(given, slice):
            sub_cpd = CPD(self.by_m, self.by_n, self.on_m, self.on_n)
            sub_cpd.by_hist = self.by_hist[given]
            sub_cpd.on_hists = self.on_hists[given]
            sub_cpd.on_null_hist = self.on_null_hist[given]
            return sub_cpd
        return self.on_hists[given]

    def __eq__(self, other):
        if self.by_hist != other.by_hist \
            or len(self.on_hists) != len(other.on_hists) \
            or self.on_null_hist != other.on_null_hist:
            return False
        for h1, h2 in zip(self.on_hists, other.on_hists):
            if h1 != h2:
                return False
        return True

    def __copy__(self):
        cpd = CPD(self.by_m, self.by_n, self.on_m, self.on_n)
        cpd.by_hist = copy.copy(self.by_hist)
        cpd.on_hists = [copy.copy(h) for h in self.on_hists]
        cpd.on_null_hist = copy.copy(self.on_null_hist)
        return cpd

    def __mul__(self, other):
        """Multiplies a CPD with a histogram."""

        if not isinstance(other, histogram.Histogram):
            raise ValueError('can only multiply a CPD with a Histogram')

        cpd = copy.copy(self)
        for i, hist in enumerate(cpd.on_hists):
            cpd.on_hists[i] = hist * other
        if cpd.on_null_hist:
            cpd.on_null_hist *= other

        return cpd

    def fit(self, by, on):
        """Fits the Histogram to `on` conditioned on `by`."""
        self.by_hist = histogram.Histogram(self.by_m, self.by_n).fit(by)

        on_values = collections.defaultdict(list)

        i = 0
        order = argsort(by)
        limit = self.by_hist[i].right

        for j in order:

            by_val, on_val = by[j], on[j]

            if by_val is None:
                on_values[None].append(on_val)
                continue

            if by_val > limit:
                i += 1
                limit = self.by_hist[i].right

            on_values[i].append(on_val)

        self.on_null_hist = histogram.Histogram(self.on_m, self.on_n)
        if None in on_values:
            self.on_null_hist = self.on_null_hist.fit(on_values.pop(None))

        self.on_hists = [
            histogram.Histogram(self.on_m, self.on_n).fit(on_values[by_val])
            for by_val in sorted(on_values)
        ]

        return self

    def p(self, by, on):
        """Returns P(on|by)."""
        i, _ = self.by_hist.find_bucket(by)
        return self.on_hists[i].p(on)

    def p_by(self, on):
        """Returns a Histogram representing P(by, on=val)"""

        hist = copy.copy(self.by_hist)

        for i, (bucket, on_hist) in enumerate(zip(self.by_hist.buckets, self.on_hists)):
            bucket.frequency = on_hist.p(on)
            bucket.cardinality = decimal.Decimal(1)
            hist.buckets[i] = bucket

        return hist

    def __str__(self):
        return '\n'.join(f'~~ {by} ~~\n{on}' for by, on in zip(self.by_hist.buckets, self.on_hists))

    def __repr__(self):
        return str(self)


def new_cpd(by_hist, on_hists, on_null_hist=None):
    """Returns a CPD with the given Histograms.

    This should only be used for testing purposes.
    """
    by_m = sum(1 for b in by_hist.buckets if b.cardinality == 1)
    by_n = sum(1 for b in by_hist.buckets if b.cardinality > 1)
    on_m = sum(1 for b in on_hists[0].buckets if b.cardinality == 1)
    on_n = sum(1 for b in on_hists[0].buckets if b.cardinality > 1)
    cpd = CPD(by_m, by_n, on_m, on_n)
    cpd.by_hist = by_hist
    cpd.on_hists = on_hists
    cpd.on_null_hist = on_null_hist
    return cpd
