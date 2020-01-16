## Use:
# (1) Generate log files from the application
#
# ttor::Communicator comm(...);
# ttor::Threadpool_mpi tp(nthread, ...);
# ttor::Logger logger(buffer_size);
# tp.set_logger(&logger);
# comm.set_logger(&logger)
# ... do things ...
# std::ofstream logfile;
# std::string filename = ...
# logfile.open(filename);
# logfile << log;
# logfile.close();
#
# Assume this creates the files file.log.0 and file.log.1 (multiple ranks)
#
# (2) Start the server and serve the profile
#
# bokeh serve --show profiler_server.py --websocket-max-message-size 200000000 --args --file file.log.*
#

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

import profiler

parser = argparse.ArgumentParser(description='Display traces output by ttor.')
parser.add_argument('--files', type=str, nargs='+', help='The files to combine & profile (at least one required)')
parser.add_argument('--start', action='store', dest='start', default=0.0, type=float, help='Only plot after start')
parser.add_argument('--end', action='store', dest='end', default=float('inf'), type=float, help='Only plot before end')
parser.add_argument('--plot_ss', action='store_true', help='Plot the stealing success lines')
parser.add_argument('--plot_comm', action='store_true', help='Plot the comm lines')
parser.add_argument('--output', action='store', dest='output', type=str, help='Save profile to file', default='profile.html')

args = parser.parse_args()

datas = [profiler.load(f) for f in args.files]
whos, data = profiler.combine(datas)
text_filter, p = profiler.make_plot(whos, data, min_start=args.start, max_end=args.end, plot_ss=args.plot_ss, plot_comm=args.plot_comm)    

layout = column(text_filter, p)
curdoc().add_root(layout)