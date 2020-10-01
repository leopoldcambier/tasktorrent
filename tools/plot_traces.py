import argparse

import ttor_logging as ttor
from bokeh.plotting import figure, show
from bokeh.io import output_file, output_notebook

# Run as
# python3 plot_traces --files log/log_ttor_cholesky.log.*
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display traces output by ttor.')
    parser.add_argument('--files', type=str, nargs='+', help='The files to combine & profile (at least one required)')
    parser.add_argument('--start', action='store', dest='start', default=0.0, type=float, help='Only plot after start')
    parser.add_argument('--end', action='store', dest='end', default=float('inf'), type=float, help='Only plot before end')
    parser.add_argument('--plot_deps', action='store_true', help='Plot the dependencies as tasks')
    parser.add_argument('--output', action='store', dest='output', type=str, help='Save profile to file', default='trace.html')
    args = parser.parse_args()

    # Example for Cholesky
    def make_color(x):
        if 'GEMM' in x:
            return 'green'
        elif 'TRSM' in x:
            return 'orange'
        elif 'POTRF' in x:
            return 'blue'
        else:
            return 'grey'

    trace, tids = ttor.make_trace(args.files, start=args.start, end=args.end, trace_deps=args.plot_deps, color=make_color)
    text, plot = ttor.make_trace_figure(trace, tids)
    output_file(args.output)
    show(plot)