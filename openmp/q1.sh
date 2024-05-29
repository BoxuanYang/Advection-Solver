#!/bin/bash

# Array of values for p
p_values=(1 2 4 8 16 32 64)

# Iterate over each value of p
for p in "${p_values[@]}"
do
    # Set the environment variable OMP_NUM_THREADS to p
    #export OMP_NUM_THREADS=$p
    echo "p is: $p --------------------------"
    
    # Run the testAdvect program with the specified arguments
    OMP_NUM_THREADS=$p ./testAdvect 1024 1024 100
done

