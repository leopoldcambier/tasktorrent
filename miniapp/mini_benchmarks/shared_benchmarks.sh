for threads in 1 4 8 12 16 20 24 28 32
do
    for sleeptime in 1000 2000 4000 8000 16000 32000 64000 128000 256000
    do
        ./mini_benchmarks ${threads} 6400 0 ${sleeptime}
    done
done

