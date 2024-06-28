#PBS -q normal
#PBS -j oe
#PBS -l walltime=00:01:00,mem=128GB
#PBS -l wd
#PBS -l ncpus=48
#

e=# echo

r=100
M=10000 # may need to make bigger
N=$M
opts="" # "-o"
ps="2 4 6 8 10"

module load openmpi


echo "" 
$e mpicc -o ts ts.c
echo mpirun -np 2 ./ts
$e mpirun -np 2 ./ts

echo "" 
$e mpicc -o tf tf.c
echo mpirun -np 2 ./tf
$e mpirun -np 2 ./tf

echo "" 
$e mpicc -o tw tw.c
echo mpirun -np 2 ./tw
$e mpirun -np 2 ./tw


if [ ! -z "$PBS_NODEFILE" ] ; then
    cat $PBS_NODEFILE
fi

exit