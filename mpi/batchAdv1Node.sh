#!/bin/bash
#PBS -q express
#PBS -j oe
#PBS -l walltime=00:01:00,mem=32GB
#PBS -l wd
#PBS -l ncpus=48
#

e= #echo

r=100
M=1000 # may need to be bigger
N=$M
opts="" # "-o" 
ps="3 6 12 24 48"

module load openmpi

for p in $ps; do
    echo ""
    echo mpirun -np $p ./testAdvect $opts $M $N $r
    $e mpirun -np $p ./testAdvect $opts $M $N $r
    echo ""
done

if [ ! -z "$PBS_NODEFILE" ] ; then
    cat $PBS_NODEFILE
fi

exit
