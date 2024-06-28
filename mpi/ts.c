#include <assert.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv){
    /// mpicc -o ts ts.c
    // mpirun -np 2 ./ts
    int rank; 
    int size;
    int tag = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        double start = MPI_Wtime();
        for(int i = 0; i < 10000; i++){
            MPI_Send(NULL, 0, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD);
        }
        double end = MPI_Wtime();
        double ts  = (end - start);
        printf("ts * 10000 for send is: %f\n", ts);
    }else if(rank == 1){
        double start = MPI_Wtime();
        for(int i = 0; i < 10000; i++){
            MPI_Recv(NULL, 0, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        double end = MPI_Wtime();
        double ts  = (end - start);
        printf("ts * 10000 for receive is: %f\n", ts);
    }
    MPI_Finalize();
}