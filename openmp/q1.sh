#!/bin/bash
#PBS -q express
#PBS -j oe
#PBS -l walltime=00:01:00,mem=32GB
#PBS -l wd
#PBS -l ncpus=4
#

# Array of values for p
p_values=(1024 2048 4096 8192)

# Iterate over each value of p
for m in "${p_values[@]}"
do
    # Set the environment variable OMP_NUM_THREADS to p
    #export OMP_NUM_THREADS=$p
    echo "M is: $m --------------------------"
    
    # Run the testAdvect program with the specified arguments
    OMP_NUM_THREADS=16 ./testAdvect $m $m 5
done


