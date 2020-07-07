import subprocess
import os

repeat = 25
n_rows = 32
for threads in [1,2,4,8,16]:
    for n_edges in [1,2,4,8,16]:
        for time in [1e-3,1e-4]:
            n_cols = round(1.0 * threads / (n_rows * time))
            subprocess.run(["./ttor_deps", str(threads), str(n_rows), str(n_edges), str(n_cols), str(time), str(repeat), "0"])
            os.environ['STARPU_NCPU'] = str(threads)
            subprocess.run(["./starpu_deps", str(n_rows), str(n_edges), str(n_cols), str(time), str(repeat), "0"])
            subprocess.run(["./starpu_deps_stf", str(n_rows), str(n_edges), str(n_cols), str(time), str(repeat), "0"])
