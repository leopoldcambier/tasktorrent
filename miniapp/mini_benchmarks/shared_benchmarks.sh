sleeptime=(8000 16000 32000 64000 128000 256000 512000 1024000)
ntasks=(64000 64000 64000 64000 64000 64000 64000 64000 64000 64000)

for threads in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
do
    for i in 0 1 2 3 4 5 6 7
    do
        ./mini_benchmarks ${threads} 6400 0 ${sleeptime[i]}
    done
done

