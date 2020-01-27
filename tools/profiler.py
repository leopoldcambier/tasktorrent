import numpy as np
from bisect import bisect_left
import argparse
import pandas as pd

from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, HoverTool, Legend, LegendItem, GlyphRenderer, Select, Slider
from bokeh.models.widgets import TextInput
from bokeh.models.glyphs import Quad, MultiLine
from bokeh.models.widgets import Dropdown
from bokeh.plotting import curdoc
from bokeh.layouts import column

def load(file):
    with open(file,'r') as f:
        data = []
        print("Loading {}".format(file))
        for l in f.readlines():
            s = l.split(",")
            info = s[0]
            start = float(s[1])
            end = float(s[2])
            ss = info.split(">")
            who = ss[0]
            what = ss[1]
            details = ss[2:]
            data.append({'who':who, 'what':what, 'details':details, 'start':start, 'end':end})
    return data

def combine(datas):
    whos = set()
    for data in datas:
        for d in data:
            whos.add(d['who'])
            if(d['what'] == 'ss' or d['what'] == 'qrpc' or d['what'] == 'lpc'):
                whos.add(d['details'][0])
    whos = list(whos)
    whos.sort()
    newdata = []
    for data in datas:
        for d in data:
            newdata.append(d)
    return whos, newdata

def find_sorted(a, x):
    i = bisect_left(a, x)
    assert(i != len(a) and a[i] == x)
    return i   

def make_plot(whos, data, min_start = 0, max_end = 1e100, plot_ss=False, plot_comm=False):
    origin = float(1e100)
    for d in data:
        origin = min(origin, d['start'])
    tooltips = [
        ("Task:", "@name"),
        ("Start:", "@start"),
        ("Duration:", "@duration")
    ]
    p = figure(width=1500, height=800, y_range=whos, tooltips=tooltips)
    p.xaxis.axis_label = "Time [s.]"
    p.yaxis.axis_label = "Thread/rank"
    rect_data = {'left':[], 'right':[], 'top':[], 'bottom':[], 'color':[], 'name':[], 'what':[], 'start':[], 'duration':[], 'alpha':[]}
    line_data = {'xs':[], 'ys':[], 'name':[]}
    comm_data = {'xs':[], 'ys':[], 'name':[]}
    extra_lines = {}

    ## Find all muuid_to_task
    muuid_to_task = {}
    for (i,d) in enumerate(data):
        who     = d['who']
        what    = d['what']
        details = d['details']
        if what == "qrpc":
            muuid = details[1]
            task = details[2]
            orig_uuid = (who,muuid)
            muuid_to_task[orig_uuid] = task

    ## Get the rest
    for (i,d) in enumerate(data):
        who      = d['who']
        what     = d['what']
        start    = d['start'] - origin
        end      = d['end']   - origin
        duration = end - start
        details  = d['details']
        tid      = find_sorted(whos, who)
        midx     = (start + end)/2
        midy     = (tid + 0.5)
        bottom   = tid+0.1
        top      = tid+0.9
        if(start >= min_start and end <= max_end):

            # Rectangles
            if(what == 'run' or what == 'sa' or what == 'deps' or what == 'idle' or what == 'qrpc' or what == 'lpc'):
                if what == "run":                
                    name = "run_" + details[0]
                    color = 'red'                                
                elif what == "sa":
                    name = 'sa'
                    color = 'green'            
                elif what == "deps":
                    name = "deps_" + details[0]
                    color = 'grey'
                elif what == "idle":
                    name = 'idle'
                    color = 'white'                
                elif what == "qrpc":
                    rpc_to = details[0]
                    muuid = details[1]
                    task = details[2]
                    name = 'qrpc_' + muuid + "_" + task
                    color = 'orange'
                    tid_rpc_to = find_sorted(whos, rpc_to)                    
                    orig_uuid = (who,muuid)
                    muuid_to_task[orig_uuid] = task
                    if orig_uuid in extra_lines:
                        extra_lines[orig_uuid]['x0'] = start
                        extra_lines[orig_uuid]['t0'] = tid
                    else:
                        extra_lines[orig_uuid] = {'x0':start, 't0':tid}
                elif what == "lpc":
                    lpc_from = details[0]
                    muuid = details[1]
                    color = 'blue'
                    tid_lpc_from = find_sorted(whos, lpc_from)
                    orig_uuid = (lpc_from,muuid)
                    task = muuid_to_task[orig_uuid]
                    name = 'lpc_' + muuid + "_" + task                    
                    if orig_uuid in extra_lines:
                        extra_lines[orig_uuid]['x1'] = end
                        extra_lines[orig_uuid]['t1'] = tid
                    else:
                        extra_lines[orig_uuid] = {'x1':end, 't1':tid}

                if details[0].startswith('dep_map_intern'):
                    color = 'black'

                rect_data['left'].append(start)
                rect_data['right'].append(end)
                rect_data['top'].append(top)
                rect_data['bottom'].append(bottom)
                rect_data['color'].append(color)
                rect_data['name'].append(name)
                rect_data['what'].append(what)
                rect_data['start'].append(start)
                rect_data['duration'].append(duration)
                rect_data['alpha'].append(1.0)

            # Lines
            if what == "ss":
                kind = 'line'
                stolen_from = details[0]
                tid_stolen_from = find_sorted(whos, stolen_from)
                top = tid+0.5
                bottom = tid_stolen_from+0.5
                line_data['xs'].append([start,end])                
                line_data['ys'].append([bottom,top])
                line_data['name'].append('ss')

    ## Extra lines between matching LPCs/RPCs
    for orig_uuid, l in extra_lines.items():
        if 'x0' in l and 'x1' in l:
            if(l['t0'] == l['t1']):
                print("Warning: message {} has same origin and destination".format(orig_uuid))
            comm_data['xs'].append([l['x0'],l['x1']])
            comm_data['ys'].append([l['t0']+0.5,l['t1']+0.5])
            comm_data['name'].append('MPIcomm')
        else:
            print("Warning: message {} was not matched".format(orig_uuid))

    ## Rectangles
    rect_data = pd.DataFrame(rect_data)
    source = ColumnDataSource(rect_data)
    r = p.add_glyph(source, Quad(left='left', right='right', top='top', bottom='bottom', fill_color='color', line_color='color', fill_alpha='alpha', line_alpha='alpha'))
    
    ## Lines
    if(plot_ss):
        r = p.add_glyph(ColumnDataSource(line_data), MultiLine(xs='xs',ys='ys'))

    if(plot_comm):
        r = p.add_glyph(ColumnDataSource(comm_data), MultiLine(xs='xs',ys='ys'))

    ## Create filter
    text_input = TextInput(value="", title="Filter")
    def text_input_fn(attr, old, new):
        fil = text_input.value
        print("Filtering using {}, originally {} rows".format(fil, rect_data.shape[0]))
        rect_data.loc[  rect_data['name'].str.contains(fil),'alpha'] = 1.0
        rect_data.loc[~ rect_data['name'].str.contains(fil),'alpha'] = 0
        source.data = rect_data.to_dict(orient='list')
        print("Done filtering")
    text_input.on_change('value', text_input_fn)

    print("Done preparing plot...")
    return text_input, p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display traces output by ttor.')
    parser.add_argument('--files', type=str, nargs='+', help='The files to combine & profile (at least one required)')
    parser.add_argument('--start', action='store', dest='start', default=0.0, type=float, help='Only plot after start')
    parser.add_argument('--end', action='store', dest='end', default=float('inf'), type=float, help='Only plot before end')
    parser.add_argument('--plot_ss', action='store_true', help='Plot the stealing success lines')
    parser.add_argument('--plot_comm', action='store_true', help='Plot the comm lines')
    parser.add_argument('--output', action='store', dest='output', type=str, help='Save profile to file', default='profile.html')

    args = parser.parse_args()

    datas = [load(f) for f in args.files]
    whos, data = combine(datas)
    text_filter, p = make_plot(whos, data, min_start=args.start, max_end=args.end, plot_ss=args.plot_ss, plot_comm=args.plot_comm)
    output_file(args.output)
    show(p)
