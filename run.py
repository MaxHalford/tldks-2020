import collections
import csv
import glob
import itertools
import os
import time

import moz_sql_parser as msp
import networkx as nx
import pandas as pd
import sqlalchemy
from sqlalchemy import orm
import tqdm


def parse_tree(query_tree):

    edges = []
    wheres = collections.defaultdict(list)
    joins = {}
    aliases = {pair['name'].lower(): pair['value'].lower() for pair in query_tree['from']}

    for where in query_tree['where']['and']:
        for op, args in where.items():
            if op == 'or':
                relation, attribute = list(args[0].values())[0][0].split('.')
                wheres[relation].append(where)
            elif op == 'between':
                relation, attribute = args[0].split('.')
                wheres[relation].append({
                    'and': [
                        {'gte': [args[0], args[1]]},
                        {'lte': [args[0], args[2]]}
                    ]
                })
            elif op in ('missing', 'exists'):
                relation, attribute = args.split('.')
                wheres[relation].append(where)
            elif op != 'eq' or not isinstance(args[1], str) or '.' not in args[1]:
                relation, attribute = args[0].split('.')
                wheres[relation].append(where)
            else:
                l_rel, l_att = args[0].split('.')
                r_rel, r_att = args[1].split('.')
                edges.append((l_rel, r_rel))
                joins[tuple(sorted([l_rel, r_rel]))] = where

    return edges, wheres, joins, aliases


def parse_query_into_graph(query):

    query = query.replace('IS NOT NULL', 'IS (NOT NULL)')

    query_tree = msp.parse(query)

    edges, wheres, joins, aliases = parse_tree(query_tree)

    graph = nx.Graph()
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, name='alias', values=aliases)
    nx.set_node_attributes(graph, name='wheres', values=wheres)
    nx.set_edge_attributes(graph, name='joins', values=joins)

    return graph


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1, 2) (1, 3) (2, 3) (1, 2, 3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))


def yield_queries(graph):

    for relations in powerset(graph.nodes):

        # Filter out the combinations that aren't joined
        sub = graph.subgraph(relations)
        if not nx.is_connected(sub):
            continue

        joins = [sub.edges[edge]['joins'] for edge in sub.edges]

        relevant_wheres = [where for relation in sub.nodes for where in graph.node[relation].get('wheres', {})]

        for wheres in powerset(relevant_wheres):

            tree = {
                'select': ['*'],
                'from': [
                    {'value': graph.node[relation]['alias'], 'name': relation}
                    for relation in relations
                ],
                'where': {
                    'and': joins + list(wheres)
                }
            }

            yield msp.format(tree), relations, joins, wheres


if __name__ == '__main__':

    uri = 'postgresql://postgres:postgres@localhost:5432/job'
    engine = sqlalchemy.create_engine(uri)
    session = orm.sessionmaker(bind=engine)()
    session.execute(f'SET statement_timeout = "{15 * 60}s"')

    if os.path.exists('results.csv'):
        done_queries = set(pd.read_csv('results.csv')['mother_query_name'])
    else:
        done_queries = set()

    with tqdm.tqdm() as pbar:

        if not os.path.exists('results.csv'):

            output = open('results.csv', 'w')
            writer = csv.DictWriter(output, fieldnames=[
                'mother_query_name',
                'sql',
                'relations',
                'joins',
                'wheres',
                'true_cardinality',
                'postgres_estimate',
                'postgres_optimization_time',
                'execution_time'
            ])
            writer.writeheader()
            output.close()

        for i, file in enumerate(sorted(glob.glob('job/join-order-benchmark/*.sql'))):

            query_name = os.path.basename(file).split('.')[0]

            # Filter out the SQL files that are not queries
            if query_name in ['fkindexes', 'foreign_keys', '11a', '11b', '11c', '12b', '13b', '13c', '14a', '14b', '15a', '15b', '15c', '16a', '19a', '19b', '19c', '19d', '1a', '1b', '1c', '1d',
                '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b', '22c', '22d']:
                continue

            # Skip if this query has already been processed
            if query_name in done_queries:
                continue

            # Show which query is being processed
            pbar.set_postfix(file=file)

            # Load the query
            query = open(file).read().rstrip().rstrip(';')

            # Convert the query to a graph
            graph = parse_query_into_graph(query)

            # Loop over each induced subquery
            query_stream = tqdm.tqdm(yield_queries(graph), unit='query')
            for sql, relations, joins, wheres in query_stream:

                try:

                    opt_tic = time.time()
                    pg_estimate = float(session.execute(f'EXPLAIN {sql}').first()[0].split('rows=')[1].split(' ')[0])
                    opt_toc = time.time()

                    exec_tic = time.time()
                    truth = session.execute(sql.replace('*', 'COUNT(*)')).first()[0]
                    exec_toc = time.time()

                    # Store the results
                    with open('results.csv', 'a') as output:
                        writer = csv.DictWriter(output, fieldnames=[
                            'mother_query_name',
                            'sql',
                            'relations',
                            'joins',
                            'wheres',
                            'true_cardinality',
                            'postgres_estimate',
                            'postgres_optimization_time',
                            'execution_time'
                        ])
                        writer.writerow({
                            'mother_query_name': file.split('/')[-1].split('.')[0],
                            'sql': sql,
                            'relations': relations,
                            'joins': joins,
                            'wheres': wheres,
                            'true_cardinality': truth,
                            'postgres_estimate': pg_estimate,
                            'postgres_optimization_time': opt_toc - opt_tic,
                            'execution_time': exec_toc - exec_tic
                        })

                # Catch timeouts
                except (sqlalchemy.exc.OperationalError, sqlalchemy.exc.InternalError):
                    continue

                except KeyboardInterrupt:
                    query_stream.close()
                    break
