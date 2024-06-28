#!/bin/bash
#PBS -q express
#PBS -j oe
#PBS -l walltime=00:01:00,mem=128GB
#PBS -l wd
#PBS -l ncpus=192
#

e=# echo

r=100
M=1000 # may need to make bigger
N=$M
opts="" # "-o"
ps="48 96 192"

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
