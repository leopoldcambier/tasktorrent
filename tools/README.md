# Profiling

This folder contains scripts used to profile TaskTorrent, generate task DAG and plot traces.

TaskTorrent has some capabilities to output two sorts of logging information.

## Traces
It can automatically output traces (i.e., beginning/end information for each task) using this idiom
```
Communicator comm(MPI_COMM_WORLD);
Threadpool tp(n_threads, &comm);
Logger logger(1000000);
tp.set_logger(&logger);
comm.set_logger(&logger);
// Application code
std::ofstream logfile;
string filename = "logging.log." + to_string(rank);
logfile.open(filename);
logfile << logger;
logfile.close();
``` 
This will create a file `logging.log.x` with x the rank, for every rank.
Those files can then be given for instance to `plot_traces.py` to display an execution trace.

## Task DAG
It has some basic primitives to also plot task DAGs. To do so, the code has to be instrumented to record dependencies, i.e., every edge need to be recorded
```
DepsLogger dlog(1000000);
// In a taskflow code
// ...
    dlog.add_event(make_unique<DepsEvent>(source_task_name, dest_task_name))
// ...
std::ofstream depsfile;
string depsfilename = "deps.dot." + to_string(rank);
depsfile.open(depsfilename);
depsfile << dlog;
depsfile.close();
```
This will create a file `deps.dot.x` with x the rank, for every rank. Those files can then be given to `plot_dag.py` to display the task DAG.