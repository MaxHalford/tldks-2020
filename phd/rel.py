import pandas as pd


class Relation(pd.DataFrame):

    _metadata = ['name', 'foreign_keys']

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        foreign_keys = kwargs.pop('foreign_keys', [])
        super().__init__(*args, **kwargs)
        self.name = name
        self.foreign_keys = [
            ForeignKey(from_rel=name, from_col=fk[0], to_rel=fk[1], to_col='index')
            for fk in foreign_keys
        ]

    @property
    def _constructor(self):
        return Relation

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        joined = super().join(other, on, how, lsuffix, rsuffix, sort)
        joined.name = f'{self.name}_{other.name}'
        joined.foreign_keys = self.foreign_keys + other.foreign_keys
        return joined


class ForeignKey():

    def __init__(self, from_rel, from_col, to_rel, to_col):
        self.from_rel = from_rel
        self.from_col = from_col
        self.to_rel = to_rel
        self.to_col = to_col

    def __str__(self):
        return f'{self.from_rel}.{self.from_col} -> {self.to_rel}.{self.to_col}'

    def __repr__(self):
        return str(self)
