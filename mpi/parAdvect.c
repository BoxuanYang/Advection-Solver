// parallel 2D advection solver module
// written for COMP4300/8300 Assignment 1
#include "omp_apps.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "serAdvect.h"

#define HALO_TAG 100

/*
Each process is a M_loc by N_loc size
*/
int M_loc, N_loc;  // local advection field size (excluding halo)
int M0, N0;        // local field element (0,0) is global element (M0,N0)
static int P0, Q0; // 2D process id (P0, Q0) in P x Q process grid

static int M, N, P, Q; // local store of problem parameters
static int verbosity;
static int rank, nprocs; // MPI values


// sets up parallel parameters above
void init_parallel_parameter_values(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_;
  P = P_, Q = Q_;
  verbosity = verb;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /*
  ...
  P(Q) P(Q+1) ... P(2Q - 1)
  P0   P1  P2 ... P(Q-1)
  Hence, we have below P0, Q0 calculation. P0, Q0 are the row&col index of the process
  */
  P0 = rank / Q;
  Q0 = rank % Q;


  // M0 is the index of the starting row for each process, same for N0
  M0 = (M / P) * P0;
  N0 = (N / Q) * Q0;

  


  // M_loc = M / P(if rank <= P-2), or M - M0. It means the number of rows for each process, same for N_loc
  M_loc = (P0 < P - 1) ? (M / P) : (M - M0);
  N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);


  
  
  /*
  //DEBUG
  printf("rank: %d\n", rank);
  printf("P: %d\n", P);
  printf("Q: %d\n", Q);
  printf("M: %d\n", M);
  printf("N: %d\n", N);
  printf("M_loc: %d\n", M_loc);
  printf("N_loc: %d\n", N_loc);
  printf("M0: %d\n", M0); 
  printf("N0: %d\n", N0); 
  printf("P0: %d\n", P0); 
  printf("Q0: %d\n\n", Q0); 
  */

} // initParParams()

// w is the halo width
void check_halo_size(int w) {
  /*
  Fix this so that if the above condition is violated in any process, all processes
   exit and only a single error message is printed out.
  */
  int error = 0;
  if (w > M_loc || w > N_loc) {
    error = 1;
  }
  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(error==1){
    if(rank==0){
      printf("%d: w=%d too large for %dx%d local field! Exiting...\n", rank, w,
           M_loc, N_loc);
    }
    exit(1);
  }
}

static void update_boundary(double *u, int ldu) {
  // top and bottom halo
  // note: we get the left/right neighbour's corner elements from each end

  // P means the number of processes
  if (P == 1) { // In case there is only 1 process, in this case no communication is needed
    for (int j = 1; j < N_loc + 1; j++) {
      u[j] = u[M_loc * ldu + j];
      u[(M_loc + 1) * ldu + j] = u[ldu + j];
    }
  } else {
    /* topProc: rank of the process that sits right above
       botProc: rank of the process that sits right below

       Signature of MPI_Send: 
       MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)

       Note that u array is organized s.t. u[i][j] == u[i * ldu + j]

       ldu == N + 2

       Note here that u is a pointer local to each process, each u is of size (M_loc + 2 * halo_width) * ldu(N+2)
       The index of u should be considered as the coordinates in the upper right of cardesian system.

       Halo area: u[0][1] - u[0][N_loc](for the bottom), u[M_loc + 1][1] - u[M_loc + 1][N_loc](for the top)
                  u[0][0] - u[M_loc+1][0](for the left), u[0][N_loc + 1] - u[M_loc + 1][N_loc + 1](for the right)

       u[i][j] = u[i * ldu + j], 0 <= i <= M_loc + 2 * halo_width - 1, 0 <= j <= N + 1
                  
    */
    


    /*
    // Deadlock issue
    int topProc = (rank + 1) % nprocs, botProc = (rank - 1 + nprocs) % nprocs;
    if(rank % 2 == 0){
      // Send upper row area to the top process's upper halo area
      MPI_Send(&u[M_loc * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD);
      
      // Receive from bottom process to update lower halo area
      MPI_Recv(&u[1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
      
      // Send lower row area to the bottom process's upper halo area
      MPI_Send(&u[ldu + 1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD);
      
      // Receive from top process to update upper halo area
      MPI_Recv(&u[(M_loc + 1) * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
      // Receive from bottom process to update lower halo area
      MPI_Recv(&u[1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
      
      // Send upper halo area to the top
      MPI_Send(&u[M_loc * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD);

      // Receive from top process to update upper halo area
      MPI_Recv(&u[(M_loc + 1) * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Send lower halo area to the bottom process
      MPI_Send(&u[ldu + 1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD);
    } */

    
    /* Non-blocking 
    // Non-blocking
    int topProc = (rank + 1) % nprocs, botProc = (rank - 1 + nprocs) % nprocs;
    MPI_Request req[4];
    MPI_Status stat[4];
    // Send upper halo area to the top
    MPI_Isend(&u[M_loc * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);
      
    // Receive from bottom process to update lower halo area
    MPI_Irecv(&u[1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD, &req[1]);
      
    // Send lower halo area to the bottom process
    MPI_Isend(&u[ldu + 1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD, &req[2]);
      
    // Receive from top process to update upper halo area
    MPI_Irecv(&u[(M_loc + 1) * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD,  &req[3]);   
    MPI_Waitall(4, req, stat); 

    */
   int topProc = (rank + Q + nprocs) % nprocs;
   int botProc = (rank - Q + nprocs) % nprocs;
   MPI_Request req[4];
   
   // It is almost the same as above except for the topProc and botProc calculation
   // Send upper halo area to the top
   MPI_Isend(&u[M_loc * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);
      
   // Receive from bottom process to update lower halo area
   MPI_Irecv(&u[1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD, &req[1]);
      
   // Send lower halo area to the bottom process
   MPI_Isend(&u[ldu + 1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD, &req[2]);
      
   // Receive from top process to update upper halo area
   MPI_Irecv(&u[(M_loc + 1) * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD,  &req[3]);   
   MPI_Waitall(4, req, MPI_STATUS_IGNORE); 
  } 

  // left and right sides of halo
  if (Q == 1) {
    for (int i = 0; i < M_loc + 2; i++) {
      u[i * ldu] = u[i * ldu + N_loc];
      u[i * ldu + N_loc + 1] = u[i * ldu + 1];
    }
  } else {
    // Next, we gonna do halo exchange with leftProc & rightProc
    int leftProc;
    if(rank % Q == 0){
      leftProc = rank + Q-1;
    }else{
      leftProc = rank - 1;
    }

    int rightProc;
    if(rank % Q == Q-1){
      rightProc = rank - (Q-1);
    }else{
      rightProc = rank + 1;
    }

    /*
    MPI_Type_vector(count, blocklen, stride, basetype, newtype) will create a new datatype, which consists of
    count instances of blocklen times basetype, with a space of stride in between
    */
    MPI_Request req[4];

    MPI_Datatype col_type;
    MPI_Type_vector(M_loc + 2, 1, ldu, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    // Send the rightmost column to the rightProc's left halo area
    MPI_Isend(&u[N_loc], 1, col_type, rightProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);

    // Receive from the leftProc to update the left halo area
    MPI_Irecv(&u[0], 1, col_type, leftProc, HALO_TAG, MPI_COMM_WORLD, &req[1]);

    // Send the leftmost column to the leftProc's right halo area
    MPI_Isend(&u[1], 1, col_type, leftProc, HALO_TAG, MPI_COMM_WORLD, &req[2]);

    // Receive from the rightProc to update the right halo area
    MPI_Irecv(&u[N_loc + 1], 1, col_type, rightProc, HALO_TAG, MPI_COMM_WORLD, &req[3]);

    MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    MPI_Type_free(&col_type);

    
  }

  
  
} 


// evolve advection with (u,ldu) containing the local field
void run_parallel_advection(int n_timesteps, double *u, int ldu) {
  double *v;
  int ldv = N_loc + 2;
  v = calloc(ldv * (M_loc + 2), sizeof(double));
  assert(v != NULL);
  assert(ldu == N_loc + 2);

  for (int t = 0; t < n_timesteps; t++) {
    update_boundary(u, ldu);
    

    /*
    The Halo area we are going to send are: 

    u[0][1] - u[0][N_loc](for the bottom)
    u[M_loc + 1][1] - u[M_loc + 1][N_loc](for the top)
    u[0][0] - u[M_loc+1][0](for the left) 
    u[0][N_loc + 1] - u[M_loc + 1][N_loc + 1](for the right)
    */

    // M_loc, N_loc refers to the size of local advection field.
    // u[ldu + 1] pointes to the bottom-left corner of the field, i.e., u[1][1]
    // v[ldv + 1] is the same
    // ldu & ldv == N+2
    update_advection_field(M_loc, N_loc, &u[ldu + 1], ldu, &v[ldv + 1], ldv);
    // u := v 
    copy_field(M_loc, N_loc, &v[ldv + 1], ldv, &u[ldu + 1], ldu);

    if (verbosity > 2) {
      char s[64];
      sprintf(s, "%d reps: u", t + 1);
      print_advection_field(rank, s, M_loc + 2, N_loc + 2, u, ldu);
    }
  }

  free(v);
} // parAdvect()



static void updateBoundry_with_comp_comm_overlap(double *u, int ldu, MPI_Request *req){
  assert(Q == 1);

  if (P == 1) { // In case there is only 1 process, in this case no communication is needed
    for (int j = 1; j < N_loc + 1; j++) {
      u[j] = u[M_loc * ldu + j];
      u[(M_loc + 1) * ldu + j] = u[ldu + j];
    }
  } else {
    
   int topProc = (rank + Q + nprocs) % nprocs;
   int botProc = (rank - Q + nprocs) % nprocs;
   
   // It is almost the same as above except for the topProc and botProc calculation

   // Receive from top process to update upper halo area
   MPI_Irecv(&u[(M_loc + 1) * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD,  &req[3]);


   // Receive from bottom process to update lower halo area
   MPI_Irecv(&u[1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD, &req[1]);

   // Send upper halo area to the top
   MPI_Isend(&u[M_loc * ldu + 1], N_loc, MPI_DOUBLE, topProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);      
      
   // Send lower halo area to the bottom process
   MPI_Isend(&u[ldu + 1], N_loc, MPI_DOUBLE, botProc, HALO_TAG, MPI_COMM_WORLD, &req[2]);
      
      
   // MPI_Waitall(4, req, MPI_STATUS_IGNORE);
  } 

  // left and right sides of halo
  for (int i = 0; i < M_loc + 2; i++) {
    u[i * ldu] = u[i * ldu + N_loc];
    u[i * ldu + N_loc + 1] = u[i * ldu + 1];
  }
  

}




// overlap communication variant
void run_parallel_advection_with_comp_comm_overlap(int reps, double *u, int ldu) {
  /*
  Implementation idea:
  In the original run_parallel_advection, we first do synchronization with all 4 
  neighbouring process, then proceed to update the local advection field.

  In this function, we will first compute boundry points, then update, finally
  compute the rest.

  We implement this only for Q==1.
  */

  // u[i][j] == u[i * ldu + j]
  assert(Q == 1);

  double *v;
  int ldv = N_loc + 2;
  v = calloc(ldv * (M_loc + 2), sizeof(double));
  assert(v != NULL);
  assert(ldu == N_loc + 2);
  
  MPI_Request req[4];

  for (int t = 0; t < reps; t++) {
    updateBoundry_with_comp_comm_overlap(u, ldu, req);
    // update inner area first

    /*
    for (int i = 0; i < M_loc + 2; i++) {
      u[i * ldu] = u[i * ldu + N_loc];
      u[i * ldu + N_loc + 1] = u[i * ldu + 1];
    } */


    update_advection_field(M_loc-2, N_loc, &u[2 * ldu + 1], ldu, &v[2 * ldv + 1], ldv);
    // The above process will run in parallel
    

    // Wait for the updateBoundry_with_comp_comm_overlap function to complete
    MPI_Waitall(4, req, MPI_STATUS_IGNORE);


    //update_boundary(u, ldu);
    

    
    update_advection_field(1, N_loc, &u[ldu + 1], ldu, &v[ldv + 1], ldv);
    update_advection_field(1, N_loc, &u[(M_loc) * ldu + 1], ldu, &v[(M_loc) * ldv + 1], ldv);

    // u := v 
    copy_field(M_loc, N_loc, &v[ldv + 1], ldv, &u[ldu + 1], ldu);

    if (verbosity > 2) {
      char s[64];
      sprintf(s, "%d reps: u", t + 1);
      print_advection_field(rank, s, M_loc + 2, N_loc + 2, u, ldu);
    }
  }

  free(v);
  
  return;

} // parAdvectOverlap()






static void update_boundary_with_wide_halo(double *u, int ldu, int w){
  if (P == 1) { // In case there is only 1 process, in this case no communication is needed
    for (int k = 0; k < w; k++){
      for (int j = w; j < N_loc + w; j++){
        u[k * ldu + j] = u[(M_loc + k) * ldu + j];
        u[(M_loc + w + k) * ldu + j] = u[(w + k) * ldu + j];
      }
    }
  } else {
    MPI_Datatype wide_row_type;
    
    MPI_Type_vector(w, N_loc, N_loc + 2 * w, MPI_DOUBLE, &wide_row_type);
    MPI_Type_commit(&wide_row_type);

    int topProc = (rank + Q + nprocs) % nprocs;
    int botProc = (rank - Q + nprocs) % nprocs;

    MPI_Request req[4];
    
    /*
    upper boundary of upper halo area
    (M_loc + 2 * w - 1, 0) ......................   (M_loc + 2 * w - 1, N_loc + 2 * w - 1)
    
    ..................(w + M_loc - 1, w) ............  (w + M_loc - 1, w + N_loc - 1).......
    ..................(w,w)             .............  (w, w + N_loc - 1)........... (local advection field)


    (0,0) ......................................... (0, N_loc + 2 * w - 1) (lower boundary of halo)
    */
   
    // Send to the top
    MPI_Isend(&u[M_loc * ldu + w],       1, wide_row_type, topProc, HALO_TAG, MPI_COMM_WORLD, &req[0]);

    // Receive from bottom, update lower halo area
    MPI_Irecv(&u[w],                     1, wide_row_type, botProc, HALO_TAG, MPI_COMM_WORLD, &req[1]);

    // Send to the bottom
    MPI_Isend(&u[w * ldu + w],           1, wide_row_type, botProc, HALO_TAG, MPI_COMM_WORLD, &req[2]);

    // Receive from the top, update upper halo area
    MPI_Irecv(&u[(M_loc + w) * ldu + w], 1, wide_row_type, topProc, HALO_TAG, MPI_COMM_WORLD, &req[3]);

   
    MPI_Waitall(4, req, MPI_STATUS_IGNORE); 
    MPI_Type_free(&wide_row_type);
  } 

  // left and right sides of halo
  if (Q == 1) {
    for (int j = 0; j < M_loc + 2 * w; j++){
      for (int i = 0; i < w; i++){
        u[j * ldu + i] = u[j * ldu + N_loc + i];
        u[j * ldu + N_loc + w + i] = u[j * ldu + w + i];
      }
    }
  } else {
    // Next, we gonna do halo exchange with leftProc & rightProc
    int leftProc;
    if(rank % Q == 0){
      leftProc = rank + Q-1;
    }else{
      leftProc = rank - 1;
    }

    int rightProc;
    if(rank % Q == Q-1){
      rightProc = rank - (Q-1);
    }else{
      rightProc = rank + 1;
    } 

    MPI_Datatype wide_column_type;
    MPI_Type_vector(M_loc + 2 * w, w, N_loc + 2 * w, MPI_DOUBLE, &wide_column_type);
    MPI_Type_commit(&wide_column_type);

    MPI_Request reqs[4];

    // Send to the right
    MPI_Isend(&u[N_loc],     1, wide_column_type, rightProc, HALO_TAG, MPI_COMM_WORLD, &reqs[0]);

    // Receive from the left, update left halo area
    MPI_Irecv(&u[0],         1, wide_column_type, leftProc,  HALO_TAG, MPI_COMM_WORLD, &reqs[1]);

    // Send to the left
    MPI_Isend(&u[w],         1, wide_column_type, leftProc,  HALO_TAG, MPI_COMM_WORLD, &reqs[2]);

    // Receive from the right, update right halo area
    MPI_Irecv(&u[N_loc + w], 1, wide_column_type, rightProc, HALO_TAG, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUS_IGNORE);

    MPI_Type_free(&wide_column_type);
  }
}



// wide halo variant
void run_parallel_advection_with_wide_halos(int reps, int w, double *u, int ldu) {
  double *v;
  int ldv = N_loc + 2 * w;
  v = calloc(ldv * (M_loc + 2 * w), sizeof(double));
  assert(v != NULL);
  assert(ldu == N_loc + 2 * w);

  for (int t = 0; t < reps; t++) {

    // for every w updates, we need to perform one boundary update
    update_boundary_with_wide_halo(u, ldu, w);
    for(int i = 1; i <= w - 1 && t < reps; i++){
      // update_advection_field(int m, int n, double *u, int ldu, double *v, int ldv)
      update_advection_field(M_loc + 2 * (w - i), N_loc + 2 * (w - i), &u[i * ldu + i], ldu, &v[i * ldv + i], ldv);
      // copy_field(int m, int n, double *in, int ldin, double *out, int ldout) 
      copy_field(M_loc + 2 * (w - i), N_loc + 2 * (w - i), &v[i * ldu + i], ldv, &u[i * ldu + i], ldu);

      
      if(t < reps){
        update_advection_field(M_loc, N_loc, &u[w * ldu + w], ldu, &v[w * ldu + w], ldv);
        copy_field(M_loc, N_loc, &v[w * ldu + w], ldv, &u[w * ldu + w], ldu);
      }

      t++;
    }

    if (verbosity > 2) {
      char s[64];
      sprintf(s, "%d reps: u", t + 1);
      print_advection_field(rank, s, M_loc + 2, N_loc + 2, u, ldu);
    }
  }
  free(v);
} // parAdvectWide()

// extra optimization variant
void run_parallel_advection_with_extra_opts(int r, double *u, int ldu) {
  double *v;
  int ldv = N_loc + 2 * w;
  v = calloc(ldv * (M_loc + 2 * w), sizeof(double));
  assert(v != NULL);
  assert(ldu == N_loc + 2 * w);

  for (int t = 0; t < reps; t++) {

    // for every w updates, we need to perform one boundary update
    update_boundary_with_wide_halo(u, ldu, w);
    for(int i = 1; i <= w - 1 && t < reps; i++){
      // update_advection_field(int m, int n, double *u, int ldu, double *v, int ldv)
      update_advection_field(M_loc + 2 * (w - i), N_loc + 2 * (w - i), &u[i * ldu + i], ldu, &v[i * ldv + i], ldv);
      // copy_field(int m, int n, double *in, int ldin, double *out, int ldout) 
      copy_field_openmp(M_loc + 2 * (w - i), N_loc + 2 * (w - i), &v[i * ldu + i], ldv, &u[i * ldu + i], ldu);

      
      if(t < reps){
        update_advection_field(M_loc, N_loc, &u[w * ldu + w], ldu, &v[w * ldu + w], ldv);
        copy_field_openmp(M_loc, N_loc, &v[w * ldu + w], ldv, &u[w * ldu + w], ldu);
      }

      t++;
    }

    if (verbosity > 2) {
      char s[64];
      sprintf(s, "%d reps: u", t + 1);
      print_advection_field(rank, s, M_loc + 2, N_loc + 2, u, ldu);
    }
  }
  free(v);
} // parAdvectExtra()



void copy_field_openmp(int m, int n, double *in, int ldin, double *out, int ldout) {
  #pragma omp parallel {

  int thread_num = omp_get_num_threads();
  int thread_id = omp_get_thread_num();

  int len;
  if(thread_id != thread_num-1){
    len = m / thread_num;
  }else{
    len =  m - (m / thread_num) * thread_num;
  }

  int start = thread_id * (m / thread_num);
  int end = start + len;
  for (int i = start; i < end; i++){
    for (int j = 0; j < n; j++){
      out[i * ldout + j] = in[i * ldin + j];
    }
  }
  }
}