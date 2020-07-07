import subprocess
import os

repeat = 25
for threads in [1, 2, 4, 8, 16]:
    for time in [1e-5, 1e-4]:
        tasks = round(threads * 1.0 / time)
        subprocess.run(["./ttor_wait", str(threads), str(tasks), str(time), "1", str(repeat), "0"])
        os.environ['OMP_NUM_THREADS'] = str(threads)
        os.environ['STARPU_NCPU'] = str(threads)
        subprocess.run(["./omp_wait", str(tasks), str(time), str(repeat), "0"])
        subprocess.run(["./starpu_wait", str(tasks), str(time), str(repeat), "0"])
        subprocess.run(["./starpu_wait_stf", str(tasks), str(time), str(repeat), "0"])
