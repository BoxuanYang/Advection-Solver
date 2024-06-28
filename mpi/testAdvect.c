// parallel 2D advection solver test program
// COMP4300/8300 Assignment 1

#include <assert.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()

#include "parAdvect.h"
#include "serAdvect.h"

#define USAGE "testAdvect [-P P] [-w w] [-o] [-x] [-v v] M N r"
#define DEFAULTS "P=nprocs w=r=1 v=0"

int rank, nprocs;    // MPI values
int M, N;            // advection field size
int n_timesteps = 1; // number of timesteps for the simulation
int P, Q;            // PxQ logical process grid , Q = nprocs / P
bool opt_wide_halo = false, opt_overlap = false,
     opt_extra = false; // set if -w, -o, -x specified
int halo_width = 1;     // halo width
int verbosity = 0;

// print a usage message for this program and exit with a status of 1
void show_usage_message(char *msg) {
  if (rank == 0) {
    printf("testAdvect: %s\n", msg);
    printf("\tusage: %sn\tdefault values: %s\n", USAGE, DEFAULTS);
    fflush(stdout);
  }
  exit(1);
}

void parse_command_line_arguments(int argc, char *argv[]) {
  char optchar; // option character returned my getopt()
  P = nprocs;
  while ((optchar = getopt(argc, argv, "P:w:oxv:")) != -1) {
    switch (optchar) {
    case 'P':
      if (!sscanf(optarg, "%d", &P)) // invalid integer
        show_usage_message("bad value for P");
      break;
    case 'w':
      if (!sscanf(optarg, "%d", &halo_width)) // invalid integer
        show_usage_message("bad value for w");
      opt_wide_halo = true;
      break;
    case 'v':
      if (!sscanf(optarg, "%d", &verbosity)) // invalid integer
        show_usage_message("bad value for v");
      break;
    case 'o':
      opt_overlap = true;
      break;
    case 'x':
      opt_extra = true;
      break;
    default:
      show_usage_message("unknown option");
      break;
    }
  }

  if (P == 0 || nprocs % P != 0)
    show_usage_message("number of processes must be a multiple of P");
  Q = nprocs / P;
  assert(Q > 0);

  if (optind < argc) {
    if (sscanf(argv[optind], "%d", &M) != 1)
      show_usage_message("bad value for M");
  } else
    show_usage_message("missing M");
  N = M;
  if (optind + 1 < argc)
    if (sscanf(argv[optind + 1], "%d", &N) != 1)
      show_usage_message("bad value for N");
  if (optind + 2 < argc)
    if (sscanf(argv[optind + 2], "%d", &n_timesteps) != 1)
      show_usage_message("bad value for n_timesteps");

  if (opt_overlap && halo_width != 1)
    show_usage_message("-o and -w w are incompatible");
} // getArgs()

void print_local_and_global_averages(int isMax, char *name, double total, int nlVals, int ngVals) {
  double v[1];
  if (verbosity > 0)
    printf("%d: local avg %s %s is %.3e\n", rank, isMax ? "max" : "avg", name,
           isMax ? total : (nlVals == 0 ? 0.0 : total / nlVals));
  MPI_Reduce(&total, v, 1, MPI_DOUBLE, isMax ? MPI_MAX : MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (rank == 0)
    printf("%s %s %.3e\n", isMax ? "Max" : "Avg", name,
           isMax ? v[0] : v[0] / ngVals);
}

typedef struct parParam {
  int rank, M0, M_loc, N0, N_loc;
} parParam;

#define numParParam (sizeof(parParam) / sizeof(int))

// compare par params on M0 and N0 attributes
int compare_parallel_parameter_values(const void *vp1, const void *vp2) {
  parParam *p1 = (parParam *)vp1, *p2 = (parParam *)vp2;
  if (p1->M0 < p2->M0)
    return (-1);
  else if (p1->M0 > p2->M0)
    return (+1);
  else if (p1->N0 < p2->N0)
    return (-1);
  else if (p1->N0 > p2->N0)
    return (+1);
  else
    return (0);
}

void gather_parallel_parameter_values() {
  parParam param = {rank, M0, M_loc, N0, N_loc};
  MPI_Send(&param, numParParam, MPI_INT, 0, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    parParam params[nprocs];
    int M0Prev = -1, M_locPrev = -1;
    for (int i = 0; i < nprocs; i++)
      MPI_Recv(&params[i], numParParam, MPI_INT, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    qsort(params, nprocs, sizeof(parParam), compare_parallel_parameter_values);
    printf("Global field decomposition:");
    for (int i = 0; i < nprocs; i++) {
      if (params[i].M0 != M0Prev || params[i].M_loc != M_locPrev) {
        M0Prev = params[i].M0;
        M_locPrev = params[i].M_loc;
        printf("\nrows %d..%d: ", params[i].M0,
               params[i].M0 + params[i].M_loc - 1);
      }
      printf("%d:%d..%d ", params[i].rank, params[i].N0,
             params[i].N0 + params[i].N_loc - 1);
    }
    printf("\n");
  }
} // gatherParParams()

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  parse_command_line_arguments(argc, argv);

  if (rank == 0) {
    printf("Advection of a %dx%d global field over %dx%d processes"
           " for %d steps.\n",
           M, N, P, Q, n_timesteps);
    if (opt_overlap)
      printf("Using overlap communication/computation\n");
    else if (opt_wide_halo)
      printf("Using wide halo technique, width=%d\n", halo_width);
    else if (opt_extra)
      printf("Using extra optimization methods\n");
  }

  init_advection_parameters(M, N);
  init_parallel_parameter_values(M, N, P, Q, verbosity);
  if (verbosity > 0)
    gather_parallel_parameter_values();
  check_halo_size(halo_width);

  int ldu = N_loc + 2 * halo_width;
  double *u = calloc(ldu * (M_loc + 2 * halo_width), sizeof(double));
  assert(u != NULL);

  init_advection_field(M0, N0, M_loc, N_loc, &u[halo_width * ldu + halo_width], ldu);
  if (verbosity > 1)
    print_advection_field(rank, "init u", M_loc, N_loc,
                     &u[halo_width * ldu + halo_width], ldu);

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  if (opt_overlap)
    run_parallel_advection_with_comp_comm_overlap(n_timesteps, u, ldu);
  else if (opt_wide_halo)
    run_parallel_advection_with_wide_halos(n_timesteps, halo_width, u, ldu);
  else if (opt_extra)
    run_parallel_advection_with_extra_opts(n_timesteps, u, ldu);
  else
    run_parallel_advection(n_timesteps, u, ldu);

  MPI_Barrier(MPI_COMM_WORLD); 
  double exec_time = MPI_Wtime() - start_time;

  if (rank == 0) {
    double gflops = 1.0e-09 * advection_flops_per_element * M * N * n_timesteps;
    printf("Advection time %.2es, GFLOPs rate=%.2e (per core %.2e)\n",
           exec_time, gflops / exec_time, gflops / exec_time / (P * Q));
  }

  if (verbosity > 1)
    print_advection_field(rank, "final u", M_loc + 2 * halo_width,
                     N_loc + 2 * halo_width, u, ldu);

  print_local_and_global_averages(0, "error of final field: ",
                   compute_error_advection_field(n_timesteps, M0, N0, M_loc, N_loc,
                                  &u[halo_width * ldu + halo_width], ldu),
                   M_loc * N_loc, M * N);
  print_local_and_global_averages(1, "error of final field: ",
                   compute_max_error_advection_field(n_timesteps, M0, N0, M_loc, N_loc,
                                     &u[halo_width * ldu + halo_width],
                                     ldu),
                   M_loc * N_loc, M * N);

  free(u);
  MPI_Finalize();
  return 0;
}
