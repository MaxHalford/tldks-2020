import collections
import functools
import operator

import pandas as pd
import sqlalchemy

from . import bn
from . import rel


class RecursiveBayesianNetwork():

    def __init__(self, max_rows=30000, sampling_method='SYSTEM', random_state=None):
        self.max_rows = max_rows
        self.sampling_method = sampling_method
        self.random_state = random_state

    def fit_database(self, con: sqlalchemy.engine.base.Connection):

        # Retrieve the foreign keys of each relation
        sql = '''
        SELECT
            tc.table_name AS from_rel,
            kcu.column_name AS from_col,
            ccu.table_name AS to_rel,
            ccu.column_name AS to_col
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE constraint_type = 'FOREIGN KEY' AND
              tc.table_schema = 'public'
        '''
        foreign_keys = {
            rel_name: [rel.ForeignKey(*fk) for _, fk in fks.iterrows()]
            for rel_name, fks in pd.read_sql(sql, con).groupby('from_rel')
        }

        # Retrieve the primary keys of each relation
        sql = '''
        SELECT
            tc.table_name,
            kcu.column_name
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
        WHERE constraint_type = 'PRIMARY KEY' AND
              tc.table_schema = 'public'
        '''
        primary_keys = pd.read_sql(sql, con).set_index('table_name')['column_name'].to_dict()

        # Determine the columns to model inside each table
        sql = '''
        SELECT
            tablename,
            attname
        FROM
            pg_stats
        WHERE
            pg_stats.schemaname = 'public' AND
            pg_stats.null_frac < 1
            -- AND n_distinct > 0
        '''
        columns = pd.read_sql(sql, con)\
                    .groupby('tablename')\
                    .apply(lambda g: set(g['attname']))\
                    .to_dict()

        # Discard the keys from the columns to model
        columns = {
            rel_name: [
                col for col in columns
                if col not in foreign_keys.get(rel_name, []) and col not in primary_keys[rel_name]
            ]
            for rel_name, columns in columns.items()
        }

        # Retrieve the number of rows inside each relation
        sql = f'''
            SELECT relname, reltuples
            FROM pg_class
            WHERE relname IN {tuple(columns.keys())}
        '''
        sizes = pd.read_sql(sql, con).set_index('relname')['reltuples'].to_dict()

        # Query the relations
        relation_names = list(columns.keys())
        self.bns_ = {}
        self.extensions_ = collections.defaultdict(list)

        # Continue while there isn't one Bayesian network per relation
        while len(self.bns_) != len(relation_names):

            # Determine which relations don't yet have a Bayesian network
            queue = set(relation_names) - set(self.bns_.keys())

            for name in queue:

                # Build a simple BN if there are no foreign keys
                if not foreign_keys.get(name):

                    # Build the query to retrieve the relation
                    sql = f'SELECT {", ".join(columns[name])} FROM {name}'

                    # Determine if sampling is required and if so how much
                    sampling = self.max_rows / sizes[name]

                    if sampling < 1:
                        sql += f' TABLESAMPLE {self.sampling_method} ({100. * sampling})'
                        if self.random_state:
                            sql += f' REPEATABLE ({self.random_state}'

                    self.bns_[name] = bn.BayesianNetwork().fit_sql(sql=sql, con=con)
                    continue

                # Skip if any of the related Bayesian networks hasn't been built yet
                if not all(self.bns_.get(f_key.to_rel) for f_key in foreign_keys[name]):
                    continue

            break



                # # Join the with the root attribute of each related table
                # star = relation
                # for f_key in f_keys:
                #     other_relation = relations[f_key.other_name]
                #     other_root = self.bns_[f_key.other_name].root
                #     self.extensions_[name].append(f_key.other_name)
                #     star = star.join(
                #         other=other_relation[[other_root]].add_prefix(f'{f_key.other_name}.'),
                #         on=f_key.this_attribute
                #     )

                # # Fit a Bayesian network to the star join
                # self.bns_[name] = bn.BayesianNetwork().fit(star)

        return self

    def fit(self, relations):
        relations = {r.name: r for r in relations}
        self.bns_ = {}
        self.extensions_ = collections.defaultdict(list)

        # Continue while there isn't one Bayesian network per relation
        while len(self.bns_) != len(relations):

            # Determine which relations don't yet have a Bayesian network
            queue = set(relations.keys()) - set(self.bns_.keys())

            for name in queue:

                relation = relations[name]
                f_keys = relation.foreign_keys

                # Build a simple BN if there are no foreign keys
                if not f_keys:
                    self.bns_[name] = bn.BayesianNetwork().fit(relation)
                    continue

                # Skip if any of the related Bayesian networks hasn't been built yet
                if not all(self.bns_.get(f_key.to_rel) for f_key in f_keys):
                    continue

                # Join the with the root attribute of each related table
                star = relation
                for f_key in f_keys:
                    other_relation = relations[f_key.to_rel]
                    other_root = self.bns_[f_key.to_rel].root
                    self.extensions_[name].append(f_key.to_rel)
                    star = star.join(
                        other=other_relation[[other_root]].add_prefix(f'{f_key.to_rel}.'),
                        on=f_key.from_col
                    )

                # Fit a Bayesian network to the star join
                self.bns_[name] = bn.BayesianNetwork().fit(star)

        return self

    def p(self, relation_names, **query):

        # Use a set for faster lookups
        relation_names = set(relation_names)

        # Determine which BNs to use
        bns = {name: bn.copy() for name, bn in self.bns_.items() if name in relation_names}

        # Determine which extensions can be applied
        extensions = {
            name: [r for r in related if r in relation_names]
            for name, related in self.extensions_.items()
            if name in relation_names
        }
        extensions = {name: related for name, related in extensions.items() if related}

        # Apply the extensions
        while extensions:

            for name in set(extensions.keys()):

                related = extensions[name]

                # Skip if any of the related extensions haven't been applied
                if any(extensions.get(r) for r in related):
                    continue

                # Apply the extensions
                for other in related:
                    bn = bns.pop(other).rename(lambda x: f'{other}.{x}')
                    root = bn.root
                    for child in bn.successors(root):
                        bns[name].add_node(child, **bn.node[child])
                        bns[name].add_edge(root, child)
                extensions.pop(name)

        # Format the query
        query = {k.replace('__', '.'): v for k, v in query.items()}

        # Compute and return the selectivity
        return functools.reduce(
            operator.mul,
            (bn.p(**query) for bn in bns.values()),
            1
        )
