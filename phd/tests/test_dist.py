from decimal import Decimal
import unittest

from phd import dist


class TestBucket(unittest.TestCase):

    def test_add(self):
        b1 = dist.Bucket('a', 'a', 0.1, 1)
        b2 = dist.Bucket('b', 'c', 0.2, 2)
        b3 = dist.Bucket('a', 'c', 0.3, 3)
        self.assertEqual(b1 + b2, b3)

    def test_str(self):
        self.assertEqual(str(dist.Bucket('a', 'a', 0.42, 1)), 'a: 0.42000')
        self.assertEqual(str(dist.Bucket('a', 'c', 0.1, 3)), '[a, c]: 0.10000 (3)')


class TestHistogram(unittest.TestCase):

    def test_get_one(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, 4])
        self.assertEqual(hist[0], dist.Bucket(1, 1, Decimal(3) / Decimal(7), 1))

    def test_get_slice(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, 4])
        sub_hist = dist.Histogram(2, 1)
        sub_hist.buckets = hist.buckets[:2]
        self.assertEqual(hist[:2], sub_hist)

    def test_fit(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, 4])
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist[0], dist.Bucket(1, 1, Decimal(3) / Decimal(7), 1))
        self.assertEqual(hist[1], dist.Bucket(2, 2, Decimal(2) / Decimal(7), 1))
        self.assertEqual(hist[2], dist.Bucket(3, 4, Decimal(2) / Decimal(7), 2))

    def test_p(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, 4])
        self.assertEqual(hist.p(0), 0)
        self.assertEqual(hist.p(5), 0)
        self.assertEqual(hist.p(1), Decimal(3) / Decimal(7))
        self.assertEqual(hist.p(2), Decimal(2) / Decimal(7))
        self.assertEqual(hist.p(3), Decimal(2) / Decimal(7) / Decimal(2))
        self.assertEqual(hist.p(4), Decimal(2) / Decimal(7) / Decimal(2))
        self.assertEqual(hist.p(2.5), 0)

    def test_mul(self):
        h1 = dist.new_histogram([
            dist.Bucket('a', 'a', 0.4, 1),
            dist.Bucket('b', 'b', 0.3, 1),
            dist.Bucket('c', 'd', 0.3, 2)
        ])
        h2 = dist.new_histogram([
            dist.Bucket('a', 'a', 0.5, 1),
            dist.Bucket('b', 'd', 0.1, 2),
            dist.Bucket('c', 'c', 0.4, 1)
        ])
        h3 = dist.new_histogram([
            dist.Bucket('a', 'a', 0.2, 1),
            dist.Bucket('b', 'b', 0.015, 1),
            dist.Bucket('c', 'd', 0.075, 2)
        ])
        h4 = dist.new_histogram([
            dist.Bucket('a', 'a', 0.2, 1),
            dist.Bucket('b', 'd', 0.03, 2),
            dist.Bucket('c', 'c', 0.06, 1)
        ])
        self.assertEqual(h1 * h2, h3)
        self.assertEqual(h2 * h1, h4)

    def test_str(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, 4])
        string = f'1: {Decimal(3) / Decimal(7):.5f}\n' + \
                 f'2: {Decimal(2) / Decimal(7):.5f}\n' + \
                 f'[3, 4]: {Decimal(2) / Decimal(7):.5f} (2)'
        self.assertEqual(str(hist), string)


class TestHistogramWithNulls(unittest.TestCase):

    def test_fit(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, None])
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist[0], dist.Bucket(1, 1, Decimal(3) / Decimal(7), 1))
        self.assertEqual(hist[1], dist.Bucket(2, 2, Decimal(2) / Decimal(7), 1))
        self.assertEqual(hist[2], dist.Bucket(3, 3, Decimal(1) / Decimal(7), 1))
        self.assertEqual(hist.null_frac, Decimal(1) / Decimal(7))

    def test_p(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, None])
        self.assertEqual(hist.p(0), 0)
        self.assertEqual(hist.p(5), 0)
        self.assertEqual(hist.p(1), Decimal(3) / Decimal(7))
        self.assertEqual(hist.p(2), Decimal(2) / Decimal(7))
        self.assertEqual(hist.p(3), Decimal(1) / Decimal(7))
        self.assertEqual(hist.p(None), Decimal(1) / Decimal(7))
        self.assertEqual(hist.p(2.5), 0)

    def test_str(self):
        hist = dist.Histogram(2, 1).fit([1, 1, 1, 2, 2, 3, None])
        string = f'1: {Decimal(3) / Decimal(7):.5f}\n' + \
                 f'2: {Decimal(2) / Decimal(7):.5f}\n' + \
                 f'3: {Decimal(1) / Decimal(7):.5f}\n' + \
                 f'None: {Decimal(1) / Decimal(7):.5f}'
        self.assertEqual(str(hist), string)


class TestCPD(unittest.TestCase):

    def test_fit(self):
        by = ['a', 'a', 'a', 'b', 'b', 'b']
        on = [1, 2, 3, 4, 5, 5]
        cpd = dist.CPD(2, 0, 3, 0).fit(by, on)
        self.assertEqual(len(cpd), 2)
        self.assertEqual(len(cpd[0]), 3)
        self.assertEqual(len(cpd[1]), 2)

    def test_p(self):
        by = ['b', 'b', 'b', 'c', 'c', 'c']
        on = [1, 2, 3, 4, 5, 5]
        cpd = dist.CPD(2, 0, 3, 0).fit(by, on)
        self.assertEqual(cpd.p('a', 1), 0)
        self.assertEqual(cpd.p('d', 1), 0)
        self.assertEqual(cpd.p('b', 0), 0)
        self.assertEqual(cpd.p('b', 1), Decimal(1) / Decimal(3))
        self.assertEqual(cpd.p('b', 2), Decimal(1) / Decimal(3))
        self.assertEqual(cpd.p('b', 3), Decimal(1) / Decimal(3))
        self.assertEqual(cpd.p('b', 4), 0)
        self.assertEqual(cpd.p('c', 4), Decimal(1) / Decimal(3))
        self.assertEqual(cpd.p('c', 5), Decimal(2) / Decimal(3))

    def test_mul(self):
        cpd = dist.new_cpd(
            by_hist=dist.new_histogram([
                dist.Bucket('a1', 'a1', 0.1, 1),
                dist.Bucket('a2', 'a2', 0.2, 1),
                dist.Bucket('a3', 'a3', 0.7, 1)
            ]),
            on_hists=[
                dist.new_histogram([
                    dist.Bucket('b1', 'b1', 0.1, 1),
                    dist.Bucket('b2', 'b2', 0.1, 1),
                    dist.Bucket('b3', 'b3', 0.8, 1)
                ]),
                dist.new_histogram([
                    dist.Bucket('b1', 'b1', 0.7, 1),
                    dist.Bucket('b2', 'b2', 0.3, 1),
                    dist.Bucket('b3', 'b3', 0.0, 1)
                ]),
                dist.new_histogram([
                    dist.Bucket('b1', 'b1', 0.6, 1),
                    dist.Bucket('b2', 'b2', 0.2, 1),
                    dist.Bucket('b3', 'b3', 0.2, 1)
                ])
            ]
        )

        hist = dist.new_histogram([
            dist.Bucket('b1', 'b1', 0.4, 1),
            dist.Bucket('b2', 'b2', 0.8, 1),
            dist.Bucket('b3', 'b3', 0.1, 1)
        ])

        result = dist.new_cpd(
            by_hist=dist.new_histogram([
                dist.Bucket('a1', 'a1', 0.1, 1),
                dist.Bucket('a2', 'a2', 0.2, 1),
                dist.Bucket('a3', 'a3', 0.7, 1)
            ]),
            on_hists=[
                dist.new_histogram([
                    dist.Bucket('b1', 'b1', 0.04, 1),
                    dist.Bucket('b2', 'b2', 0.08, 1),
                    dist.Bucket('b3', 'b3', 0.08, 1)
                ]),
                dist.new_histogram([
                    dist.Bucket('b1', 'b1', 0.28, 1),
                    dist.Bucket('b2', 'b2', 0.24, 1),
                    dist.Bucket('b3', 'b3', 0.00, 1)
                ]),
                dist.new_histogram([
                    dist.Bucket('b1', 'b1', 0.24, 1),
                    dist.Bucket('b2', 'b2', 0.16, 1),
                    dist.Bucket('b3', 'b3', 0.02, 1)
                ])
            ]
        )

        self.assertEqual(cpd * hist, result)


class TestCPDWithNulls(unittest.TestCase):

    def test_fit_by_nulls(self):
        by = ['a', 'a', 'a', 'b', 'b', 'b', None, None]
        on = [1, 2, 3, 4, 5, 5, 1, 2]
        cpd = dist.CPD(2, 0, 3, 0).fit(by, on)
        self.assertEqual(len(cpd), 2)
        self.assertEqual(len(cpd[0]), 3)
        self.assertEqual(len(cpd[1]), 2)
        self.assertEqual(len(cpd.on_null_hist), 2)
