#include <assert.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv){
    // mpicc -o tf tf.c
    // mpirun -np 2 ./tf
    MPI_Init(&argc, &argv);
    double x = 2.18239;
    double start_time = MPI_Wtime();
    for(int i = 0; i < 10000000; i++){
        x = x * 3.14;
        x = x * 0.38;
    }
    double end_time = MPI_Wtime();
    
    double tw = (end_time - start_time);

    printf("tf * 20000000 is: %f\n", tw);
    MPI_Finalize();
}