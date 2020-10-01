import argparse

import ttor_logging as ttor
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import curdoc

# Example: bokeh serve --show animate_dag.py --args --edges deps_ttor_dist_256_16_1_4_0.dot.0 --nodes ttor_dist_256_16_1_4_0.log.0 --show_edges
parser = argparse.ArgumentParser(description='Display animated dependency graph from ttor.')
parser.add_argument('--edges', type=str, nargs='+', help='The *.dot files')
parser.add_argument('--nodes', type=str, nargs='+', help='The *.log files')
parser.add_argument('--show_edges', action='store_true', help='Plot edges')

args = parser.parse_args()

nodes = ttor.make_DAG_nodes(args.nodes)
edges = ttor.make_DAG_edges(args.edges)

def color(x):
    if 'POTRF' in x:
        return 'red'
    elif 'TRSM' in x:
        return 'orange'
    else:
        return 'green'
nodes['color'] = nodes['what'].map(color)

plot, slider = ttor.make_DAG_figure(nodes, edges, args.show_edges)
layout = column(plot, slider)

curdoc().title = "TTOR Dynamic DAG plot"
curdoc().add_root(layout)