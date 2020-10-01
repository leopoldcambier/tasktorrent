import argparse
from graphviz import Digraph
import ttor_logging as ttor

## Example: python3 plot_dag.py --edges deps_ttor.dot.*
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display dependency graph from ttor.')
    parser.add_argument('--edges', type=str, nargs='+', help='The *dot files to combine to create the DAG')

    args = parser.parse_args()
    edges = ttor.make_DAG_edges(args.edges)

    g = Digraph('G')
    for i,r in edges.iterrows():
        g.edge(r['start'],r['end'])
    g.view()