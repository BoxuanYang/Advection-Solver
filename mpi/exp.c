#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define HALO_TAG 100

int main(int argc, char **argv){
    // mpicc -o exp exp.c
    // mpirun -np 2 ./exp
    int M_loc = 2;
    int N_loc = 4;
    int w = 3;
    int ldu = N_loc + 2 * w;
    double *u = calloc(ldu * (M_loc + 2 * w), sizeof(double));
    for(int i = 0; i < M_loc + 2 * w; i++){
        for(int j = 0; j < N_loc + 2 * w; j++){
            u[i * ldu + j] = (double) i * ldu + j;
        }
    }

    int rank, nprocs;

    


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    


    

    if(rank % 2 == 0){
        // Process 0 send to process 1
        MPI_Datatype wide_col_type;
        
        MPI_Type_vector(M_loc + 2 * w, w, N_loc + 2 * w, MPI_DOUBLE, &wide_col_type);
        
        MPI_Type_commit(&wide_col_type);

        MPI_Request req[1];

        int leftProc;
        if(rank - 1 >= 0){
            leftProc = rank - 1;
        }else{
            leftProc = nprocs-1;
        }

        printf("process %d sending to process %d\n", rank, leftProc);

        // Send left halo to the left proc
        MPI_Isend(&u[0],       1, wide_col_type, leftProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);

        MPI_Wait(req, MPI_STATUS_IGNORE);
        
        MPI_Type_free(&wide_col_type); 
        printf("Finished sending\n");
        

    }else if(rank % 2 == 1){
        MPI_Datatype wide_col_type;
        
        MPI_Type_vector(M_loc + 2 * w, w, N_loc + 2 * w, MPI_DOUBLE, &wide_col_type);
        
        MPI_Type_commit(&wide_col_type);

        MPI_Request req[1];

        double *buf = calloc(ldu * (M_loc + 2 * w), sizeof(double));

        int rightProc = (rank + 1) % nprocs;

        // receive the left halo
        MPI_Irecv(&buf[0], 1, wide_col_type, rightProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);

        printf("process %d receiving from process %d\n", rank, rightProc);

 
        


        MPI_Wait(req, MPI_STATUS_IGNORE);

        //printf("left halo: \n");
        for(int i = 0; i < M_loc + 2 * w; i++){
            for(int j = 0; j < w; j++){
                //printf("buf[%d]: %f\n", i*ldu+j, buf[i*ldu+j]);
            }
        }
        

        
        
        
        MPI_Type_free(&wide_col_type);
        printf("Finished receiving\n");

        
    }

     MPI_Finalize();

     





    
}