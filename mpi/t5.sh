#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l walltime=00:01:00,mem=128GB
#PBS -l wd
#PBS -l ncpus=96
#

e=# echo

r=100
M=1000 # may need to make bigger
N=$M
opts="-o"
ps="2 4 8 16 32 64 128"

module load openmpi

for p in $ps; do
    echo ""
    echo mpirun -np $p ./testAdvect $opts $M $N $r
    $e mpirun -np $p ./testAdvect $opts $M $N $r

    echo ""
    echo mpirun -np $p ./testAdvect $M $N $r
    $e mpirun -np $p ./testAdvect $M $N $r
    echo ""
    echo ""
done

if [ ! -z "$PBS_NODEFILE" ] ; then
    cat $PBS_NODEFILE
fi

exit