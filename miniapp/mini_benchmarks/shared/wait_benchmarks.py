import subprocess
import os

repeat = 25
for threads in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
    for time in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        tasks = round(threads * 1.0 / time)
        subprocess.run(["./ttor_wait", str(threads), str(tasks), str(time), "0", str(repeat), "0"])
