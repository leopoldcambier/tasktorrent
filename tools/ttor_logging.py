import argparse
import pandas as pd
import networkx as nx
from bisect import bisect_left

from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Slider, LinearColorMapper, CategoricalColorMapper, Quad, TextInput
from bokeh.palettes import Viridis6, Category10_3

def make_DAG_nodes(files):
    df = pd.concat([pd.read_csv(f, header=None, names=["what", "time_start","time_end"]) for f in files], axis=0)
    df = df[df['what'].str.contains(">run>")]
    df = df[~df['what'].str.contains("intern")]
    cols = df['what'].str.split(pat=">",expand=True)
    df['task'] = cols[2]
    mint = min(df['time_end'])
    df['time'] = df['time_end'] - mint
    df['kind'] = [s.split('_')[0] for s in df['task']]
    df = df.set_index('task')
    return df

def make_DAG_edges(files):
    df = pd.concat([pd.read_csv(f, header=None, names=["start","end"]) for f in files], axis=0)
    df = df.reindex()
    return df

def make_DAG(nodes, edges):
    G = nx.DiGraph()
    for index, row in nodes.iterrows():
        G.add_node(index)
    for index, row in edges.iterrows():
        G.add_edge(row['start'], row['end'])
    return G

def compute_DAG_pos(graph):
    print("Computing layout. This may take a while...")
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    print("Done with layout")
    return pos

def add_DAG_pos(nodes, edges):
    G = make_DAG(nodes, edges)
    pos = compute_DAG_pos(G)
    nodes['x'] = [pos[n][0] for n in nodes.index]
    nodes['y'] = [pos[n][1] for n in nodes.index]

def find_sorted(a, x):
    i = bisect_left(a, x)
    assert(i != len(a) and a[i] == x)
    return i

def default_pick_color(x):
    row = x.split('>')
    who = row[0]
    what = row[1]
    details = row[2:]
    if what == "run":                
        return 'red'
    elif what == "sa":
        return 'green'
    elif what == "sa":
        return 'green'
    elif what == "deps":
        return 'grey'
    elif what == "idle":
        return 'white'
    else:
        print(f"Missing condition for {what} in default_pick_color")
        return 'orange'

def default_pick_name(x):
    row = x.split('>')
    who = row[0]
    what = row[1]
    details = row[2:]
    if what == "run":
        return "run_" + details[0]
    elif what == "sa":
        return "sa"
    elif what == "ss":
        return "ss"
    elif what == "deps":
        return "deps_" + details[0]
    elif what == "idle":
        return "idle"
    else:
        print(f"Missing condition for {what} in default_pick_name")
        return 'orange'

def default_pick_alpha(x):
    return 1.0

def make_trace(files, trace_deps=False, color=default_pick_color, name=default_pick_name, alpha=default_pick_alpha, start=0.0, end=float('inf')):
    df = pd.concat([pd.read_csv(f, sep=',', names=["info", "start", "end"], header=None) for f in files], axis=0)
    mint = min(df['start'])
    df['start'] = df['start'] - mint
    df['end'] = df['end'] - mint
    df = df[df['start'] >= start]
    df = df[df['end'] <= end]

    ss = [row.split('>') for row in df['info']]
    df['who'] = [row[0] for row in ss]
    df['what'] = [row[1] for row in ss]
    df['details'] = [row[2:] for row in ss]

    if(not trace_deps):
        df = df[~df['what'].str.contains("deps")]
    
    df['duration'] = df['end'] - df['start']

    # Compute an ID for every rank/thread
    tids = list(pd.unique(df['who']))
    tids.sort()
    df['tid'] = df['who'].map(lambda x: find_sorted(tids, x))

    # Compute squares
    df['top'] = df['tid']+0.1
    df['bottom'] = df['tid']+0.9

    # Compute colors
    df['color'] = df['info'].map(color)

    # Make name
    df['name'] = df['info'].map(name)

    # Initial alpha
    df['alpha'] = df['info'].map(alpha)

    print("Done computing traces")
    return df, tids

def make_trace_figure(trace, tids):
    
    tooltips = [
        ("Task:", "@name"),
        ("Start:", "@start"),
        ("Duration:", "@duration")
    ]
    mint = min(trace['start'])
    maxt = max(trace['end'])
    plot = figure(width=1500, height=800, tooltips=tooltips, y_range=tids, x_range=(mint, maxt))
    plot.xaxis.axis_label = "Time [s.]"
    plot.yaxis.axis_label = "Thread/rank"

    source = ColumnDataSource(trace)
    rect = plot.add_glyph(source, Quad(left='start', right='end', top='top', bottom='bottom', fill_color='color', line_color='color', fill_alpha=0.9, line_alpha=0.9))

    ## Create filter
    text_input = TextInput(value="", title="Filter")
    def text_input_fn(attr, old, new):
        fil = text_input.value
        new_trace = trace[trace['name'].str.contains(fil)]
        print("Filtering using {}, originally {} rows, now {} rows".format(fil, trace.shape[0], new_trace.shape[0]))
        source.data = new_trace
        print("Done filtering")
    text_input.on_change('value', text_input_fn)

    print("Done preparing plot...")
    return text_input, plot

def make_DAG_figure(nodes, edges, show_edges=False):

    # Add positions
    if 'x' not in nodes and 'y' not in nodes:
        add_DAG_pos(nodes, edges)
    posx = nodes['x']
    posy = nodes['y']
    tt   = nodes['time']
    (minx,maxx) = (min(posx),max(posx))
    (miny,maxy) = (min(posy),max(posy))

    # Create figure
    plot = figure(width=800, height=800, title="Task graph", x_range=(minx-10,maxx+10), y_range=(miny-10,maxy+10))

    # Edges
    if show_edges:
        edges_pos = ColumnDataSource(pd.DataFrame({
            'xs':[ [posx[e['start']],posx[e['end']]] for index, e in edges.iterrows() ],
            'ys':[ [posy[e['start']],posy[e['end']]] for index, e in edges.iterrows() ],
        }))
        model_edges = plot.multi_line('xs', 'ys', source=edges_pos, color='black', line_width=1)

    # Nodes
    model_nodes = plot.circle('x', 'y', source=nodes, color='color', alpha=1.0, size=10)

    TOOLTIPS = [
        ("task", "@task"),
        ("t", "@time"),
    ]   

    # Around
    plot.add_tools(HoverTool(renderers=[model_nodes], tooltips=TOOLTIPS))
    plot.axis.visible = False
    plot.grid.visible = False
    
    # Slider
    maxt = max(tt)
    slider = Slider(start=0, end=maxt, step=maxt/25, value=maxt, title='Time (ms.)')
    def slider_callback(attr, old, new):
        new_max_t = new
        print(f"New time bound is {new_max_t}")
        model_nodes.data_source.data = nodes[nodes['time'] <= new_max_t]
        if show_edges:
            model_edges.data_source.data = pd.DataFrame({
                'xs':[ [posx[e['start']],posx[e['end']]] for index, e in edges.iterrows() if tt[e[1]] <= new_max_t ],
                'ys':[ [posy[e['start']],posy[e['end']]] for index, e in edges.iterrows() if tt[e[1]] <= new_max_t ],
            })
    slider.on_change('value', slider_callback)

    return plot, slider