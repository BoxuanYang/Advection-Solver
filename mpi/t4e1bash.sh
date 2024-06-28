#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l walltime=00:01:00,mem=128GB
#PBS -l wd
#PBS -l ncpus=48
#

e=# echo

r=100
M=1000 # may need to make bigger
N=$M
opts="" # "-o"
np=24
ps="1 2 3 4 6 8 12 24"

module load openmpi

for p in $ps; do
    echo ""
    echo mpirun -np $np ./testAdvect -P $p $opts $M $N $r
    $e mpirun -np $np ./testAdvect -P $p $opts $M $N $r
    echo ""
done


if [ ! -z "$PBS_NODEFILE" ] ; then
    cat $PBS_NODEFILE
fi

exit