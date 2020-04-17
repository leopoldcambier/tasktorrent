import subprocess

for threads in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
    for time in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        tasks = threads * 1.0 / max(time, 1e-6)
        subprocess.run(["./wait", str(threads), str(tasks), "0", str(time)])
