import numpy as np
import argparse
import pandas as pd
from graphviz import Digraph

def load(file):
    with open(file,'r') as f:
        data = []
        print("Loading {}".format(file))
        for l in f.readlines():
            s = l.strip('\n').split(",")
            n1 = s[0]
            n2 = s[1]
            data.append((n1, n2))
    return data

def combine(datas):
    data = []
    for d in datas:
        data = data + d
    return data

def display(edges):

    # t0 = 1e100
    # t1 = 0
    # for (n, nd) in nodes.items():
    #     t = float(nd['start'])
    #     t0 = min(t0, t)
    #     t1 = max(t1, t)

    g = Digraph('G')
    # for (n, nd) in nodes.items():
    #     t = float(nd['start'])
    #     s = (t - t0) / (t1 - t0)
    #     g.node(n, time = str(s))
    for e in edges:
        g.edge(e[0], e[1])
    g.view()

# def gen_nodes(datas):
#     nodes = {}
#     for data in datas:
#         for d in data:
#             if d['what'] == 'run':
#                 task = d['details'][0]
#                 nodes[task] = d
#     return nodes   

## Example: python3.7 dep_graph.py --files deps_ttor_dist_1_2_2_2_0.dot.*
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display dependency graph from ttor.')
    parser.add_argument('--files', type=str, nargs='+', help='The files to combine & plot dependency from (at least one required)')

    args = parser.parse_args()

    datas = [load(f) for f in args.files]
    edges = combine(datas)

    display(edges)

