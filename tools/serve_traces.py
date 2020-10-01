## Use:
#
# (1) Generate log files from the application
#
# ttor::Communicator comm(...);
# ttor::Threadpool tp(nthread, ...);
# ttor::Logger logger(buffer_size); # buffer_size should be big, like 1M or so
# tp.set_logger(&logger);
# comm.set_logger(&logger)
# ... do things ...
# std::ofstream logfile;
# std::string filename = ...
# logfile.open(filename);
# logfile << log;
# logfile.close();
#
# Assume this creates the files file.log.0 and file.log.1 (1 per rank)
#
# (2) Start the server and serve the profile
#
# bokeh serve --show serve_traces.py --websocket-max-message-size 200000000 --args --file file.log.*
#

import argparse

import ttor_logging as ttor
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import curdoc

parser = argparse.ArgumentParser(description='Display traces output by ttor.')
parser.add_argument('--files', type=str, nargs='+', help='The files to combine & profile (at least one required)')
parser.add_argument('--start', action='store', dest='start', default=0.0, type=float, help='Only plot after start')
parser.add_argument('--end', action='store', dest='end', default=float('inf'), type=float, help='Only plot before end')
parser.add_argument('--plot_deps', action='store_true', help='Plot the dependencies as tasks')
args = parser.parse_args()

trace, tids = ttor.make_trace(args.files, start=args.start, end=args.end, trace_deps=args.plot_deps)
text, plot = ttor.make_trace_figure(trace, tids)

layout = column(text, plot)
curdoc().add_root(layout)