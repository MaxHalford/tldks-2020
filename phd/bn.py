import decimal
import itertools

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False
import networkx as nx
import pandas as pd
from sklearn import metrics
from sklearn import utils
import sqlalchemy

from . import cpd
from . import histogram
from . import null
from . import rel


class BayesianNetwork(nx.DiGraph):

    def __init__(self, incoming_graph_data=None, cl_max_rows=30000, random_state=None,
                 unique_ratio_limit=1.0):
        super().__init__(incoming_graph_data)
        self.cl_max_rows = cl_max_rows
        self.random_state = utils.check_random_state(random_state)

    def fit(self, relation):
        """Fits the BayesianNetwork to a Relation."""

        # Drop the columns with unique values
        n_uniques = relation.nunique() # TODO, consider different n_unique ratio thresholds
        for col in relation.columns[n_uniques == len(relation)]:
            relation.drop(columns=col, inplace=True)

        # Replace nulls with a random string
        if len(relation) > self.cl_max_rows:
            r = relation.sample(n=self.cl_max_rows, random_state=self.random_state)
        else:
            r = relation.copy()

        # Replace missing values
        for col in r.select_dtypes(include=['object', 'category']):
            if r[col].isnull().sum() > 0:
                r[col].fillna('MISSING', inplace=True)
        for col in r.select_dtypes(exclude=['object', 'category']):
            if r[col].isnull().sum() > 0:
                r[col].fillna(-1, inplace=True)

        # Only keep meaningful attributes, not the foreign keys
        attributes = r.columns.drop([fk.from_col for fk in r.foreign_keys])

        # Find the structure
        cl = build_chow_liu(r[attributes])
        self = BayesianNetwork(
            incoming_graph_data=cl,
            cl_max_rows=self.cl_max_rows,
            random_state=self.random_state
        )

        return self

        # Compute the CPDs
        self.update(relation)

        return self

    def fit_sql(self, sql: str, con: sqlalchemy.engine.base.Connection):
        """Fits the BayesianNetwork to a Relation derived from an SQL query."""
        relation = rel.Relation(pd.read_sql(sql, con=con))
        return self.fit(relation)

    def update(self, relation, by_m=30, by_n=30, on_m=30, on_n=30):
        """Updates the distributions of the network.

        First, the root node is annotated with a histogram. Then, each node is
        annotated with a conditional probability distribution conditioned on it's
        parent. The tree is traversed in a depth-search manner.
        """

        if self.number_of_nodes() == 0:
            return self

        def walk(node, parent):

            by = relation[parent].fillna(null.Null()).values.tolist()
            on = relation[node].fillna(null.Null()).values.tolist()

            self.nodes[node]['dist'] = cpd.CPD(by_m, by_n, on_m, on_n).fit(by, on)

            for child in self.successors(node):
                walk(child, node)

        root = self.root
        root_values = relation[root].fillna(null.Null())
        self.nodes[root]['dist'] = histogram.Histogram(on_m, on_n).fit(root_values.values.tolist())
        for child in self.successors(root):
            walk(child, root)

        return self

    def steiner_tree(self, nodes):
        """Returns the minimal part of the tree that contains a set of nodes."""
        nodes = list(nodes)
        bunch = set()

        def walk(node, path):

            if node in nodes:
                bunch.update(path + [node])
                nodes.remove(node)

            for child in self.successors(node):
                walk(child, path + [node])

        walk(self.root, [])

        return BayesianNetwork(self.subgraph(bunch))

    def infer(self, query):
        """Returns the estimated selectivity of a query."""

        def walk(node):

            cpd = self.node[node]['dist']

            for child in self.successors(node):
                cpd *= walk(child)

            condition = query.get(node)
            if condition is not None:
                return cpd.p_by(condition)

            hist = cpd.by_hist
            for i, (bucket, on_hist) in enumerate(zip(cpd.by_hist.buckets, cpd.on_hists)):
                bucket.frequency = sum(b.frequency for b in on_hist)
                bucket.cardinality = decimal.Decimal(1)
                hist.buckets[i] = bucket
            return hist

        root = self.root
        hist = self.node[root]['dist']

        for child in self.successors(root):
            hist = hist * walk(child)

        condition = query.get(root)
        if condition is not None:
            return hist.p(condition)
        return sum(b.frequency for b in hist.buckets)

    def p(self, **query):
        """Eye candy on top of `infer`."""
        relevant = self.steiner_tree(query.keys())
        return float(relevant.infer(query))

    @property
    def root(self):
        """Returns the root node of the network."""
        return next(nx.topological_sort(self))

    def to_dot(self):
        """Returns a pydot object representing the BayesianNetwork."""
        return nx.nx_pydot.to_pydot(self)

    def draw(self):
        """Produces a GraphViz representation.

        Raises a `RuntimeError` if GraphViz is not installed.
        """
        if not GRAPHVIZ_INSTALLED:
            raise RuntimeError('graphviz needs to be installed')
        return graphviz.Source(self.to_dot())

    def rename(self, mapping):
        """Renames each node according to a mapping."""
        return BayesianNetwork(nx.relabel_nodes(self, mapping))

    def copy(self):
        """Returns an independent copy."""
        return BayesianNetwork(super().copy())


def build_chow_liu(relation):
    """Builds a tree from a relation using the Chow-Liu algorithm.

    Args:
        relation (dict): A dictionary mapping column names to a list of
            matching values. For example:

                relation = {
                    'hair': ['Blond', 'Blond', 'Brown'],
                    'nationality': ['Swedish', 'American', 'American']
                }

    Returns:
        networkx.DiGraph: A directed graph with a tree structure.

    """

    # Create a graph that contains all the mutual information values
    mut_info_graph = nx.Graph()

    for (a, b) in itertools.combinations(relation.columns, 2):
        mut_info = metrics.normalized_mutual_info_score(
            labels_true=relation[a],
            labels_pred=relation[b],
            average_method='arithmetic'
        )
        mut_info_graph.add_edge(a, b, weight=mut_info)

    # Determine the maximum spanning tree
    mst = nx.maximum_spanning_tree(mut_info_graph)
    return nx.dfs_tree(mst)
