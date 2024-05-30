#!/bin/bash
#PBS -q express
#PBS -j oe
#PBS -l walltime=00:01:00,mem=32GB
#PBS -l wd
#PBS -l ncpus=4
#

# Array of values for p
p_values=(1 2 4 8)

# Iterate over each value of p
for p in "${p_values[@]}"
do
    # Set the environment variable OMP_NUM_THREADS to p
    #export OMP_NUM_THREADS=$p
    echo "p is: $p --------------------------"
    
    # Run the testAdvect program with the specified arguments
    OMP_NUM_THREADS=8 ./testAdvect -P $p  1024 1024 100
done


