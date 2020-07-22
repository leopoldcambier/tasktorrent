
seq_t = dict()
data = []

with open("ttor_gemm_2d_raw.dat", "r") as filestream:
    for line in filestream:
        currentline = line.split(' ')
        total_time = float(currentline[11])
        num_blocks = float(currentline[9])
        n_cores = int(currentline[3])
        n_ranks = int(currentline[2])
        matrix_size = int(currentline[8])
        if (n_cores == 1):
            seq_t[matrix_size] = total_time
        if (matrix_size == 16384 or matrix_size == 8192):
            data.append([n_ranks, matrix_size, 1/(total_time * float(n_cores)), num_blocks * num_blocks / float(n_cores)])

for d in data:
    d[2] = d[2] * seq_t[d[1]]

with open("ttor_eff_conc.dat", "w") as out:
    heading = "nodes n eff conc"
    out.write(heading + '\n')
    for d in data:
        str_t = ""
        for i in range(len(d)):
            str_t += str(d[i]) + ' '
        str_t += '\n'
        out.write(str_t)


        

        
