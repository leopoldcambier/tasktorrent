import subprocess
import os

repeat = 3
n_rows = 64
for threads in [1, 2, 4, 8, 16, 24, 32]:
    for n_edges in [1, 2, 4, 8, 16, 32]:
        for time in [1e-4, 1e-3]:
            n_cols = round(1.0 * threads / (n_rows * time))
            subprocess.run(["./ttor_deps", str(threads), str(n_rows), str(n_edges), str(n_cols), str(time), str(repeat), "0"])
