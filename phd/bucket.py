import decimal


class Bucket():
    """A Bucket is a piece of a Histogram."""

    def __init__(self, left, right, frequency, cardinality):
        self.left = left
        self.right = right
        if not isinstance(frequency, decimal.Decimal):
            frequency = decimal.Decimal(str(frequency))
        self.frequency = frequency
        if not isinstance(cardinality, decimal.Decimal):
            cardinality = decimal.Decimal(str(cardinality))
        self.cardinality = cardinality

    def __copy__(self):
        return Bucket(self.left, self.right, self.frequency, self.cardinality)

    def __add__(self, other):
        """[a, b] + [b, c] = [a, c]"""
        return Bucket(
            left=min(self.left, other.left),
            right=max(self.right, other.right),
            frequency=self.frequency + other.frequency,
            cardinality=self.cardinality + other.cardinality
        )

    def __eq__(self, other):
        """[a, b] == [a, b]"""
        return self.left == other.left and \
            self.right == other.right and \
            self.frequency == other.frequency and \
            self.cardinality == other.cardinality

    def __gt__(self, other):
        """[b, c] > a"""
        return self.left > other

    def __lt__(self, other):
        """[a, b] < c"""
        return self.right < other

    def __str__(self):
        if self.cardinality == 1:
            return f'{self.left}: {self.frequency:.5f}'
        return f'[{self.left}, {self.right}]: {self.frequency:.5f} ({self.cardinality})'

    def __repr__(self):
        return str(self)
