// serial 2D advection solver module
// COMP4300/8300 Assignment 1

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "serAdvect.h"


// advection parameters
const double CFL = 0.25;             // CFL condition number
const double Velx = 1.0, Vely = 1.0; // advection velocity
double dt;                           // time for 1 step
double deltax, deltay;               // grid spacing//

void init_advection_parameters(int M, int N) {
  assert(M > 0 && N > 0); // advection not defined for empty grids
  deltax = 1.0 / N;
  deltay = 1.0 / M;
  dt = CFL * (deltax < deltay ? deltax : deltay);
}

/*
Evaluate the initial condition of x, y, 
*/
double evaluate_initial_condition(double x, double y, double t) {
  x = x - Velx * t;
  y = y - Vely * t;
  return (sin(4.0 * M_PI * x) * sin(2.0 * M_PI * y));
}

/*
u is a one dimensional array where u[i * ldu + j] == u[i][j] in 2D dimention

Set all u to initial condition
*/
void init_advection_field(int M0, int N0, int m, int n, double *u, int ldu) {
  for (int i = 0; i < m; i++) {
    double y = deltay * (i + M0);
    for (int j = 0; j < n; j++) {
      double x = deltax * (j + N0);
      u[i * ldu + j] = evaluate_initial_condition(x, y, 0.0);
    }
  }
}



double compute_error_advection_field(int timestep, int M0, int N0, int m, int n, double *u, int ldu) {
  double err = 0.0;
  double t = timestep * dt;
  for (int i = 0; i < m; i++) {
    double y = deltay * (i + M0);
    for (int j = 0; j < n; j++) {
      double x = deltax * (j + N0);
      err += fabs(u[i * ldu + j] - evaluate_initial_condition(x, y, t));
    }
  }
  return (err);
}

double compute_max_error_advection_field(int r, int M0, int N0, int m, int n, double *u,
                         int ldu) {
  double err = 0.0;
  double t = r * dt;
  for (int j = 0; j < n; j++) {
    double x = deltax * (j + N0);
    for (int i = 0; i < m; i++) {
      double y = deltay * (i + M0);
      double e = fabs(u[i * ldu + j] - evaluate_initial_condition(x, y, t));
      if (e > err)
        err = e;
    }
  }
  return (err);
}

const int advection_flops_per_element = 20; // count 'em

/**
 * v : Ux or Uy, 位移
 * cm1 = v/2 * (v + 1)
 * c0  = 1 - v^2
 * cp1 = v/2 * (v-1) = v^2/2 - v/2 
 * 
*/
void calculate_and_update_coefficients(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v / 2.0;
  *cm1 = v2 * (v + 1.0);
  *c0 = 1.0 - v * v;
  *cp1 = v2 * (v - 1.0);
}

// uses the Lax-Wendroff method
/*
u is the old u
v is the new u

cim1 * (cjm1 cj0 cjp1)
ci0  * (cjm1 cj0 cjp1)
cip1 * (cjm1 cj0 cjp1)

ldu = N + 2
*/
void update_advection_field(int m, int n, double *u, int ldu, double *v, int ldv) {
  // Ux, Uy means unit offset, i.e., v * t
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1;
  double cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  //printf("ldu: %d\n", ldu);
  //printf("ldv: %d\n", ldv);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);

}


/**/
void print_advection_field(int rank, char *label, int m, int n, double *u, int ldu) {
  printf("%d: %s\n", rank, label);
  for (int i = 0; i < m; i++) {
    printf("%d: ", rank);
    for (int j = 0; j < n; j++)
      printf(" %+0.2f", u[i * ldu + j]);
    printf("\n");
  }
}

/*
Copy the data in array out into array in
*/
void copy_field(int m, int n, double *in, int ldin, double *out, int ldout) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      out[i * ldout + j] = in[i * ldin + j];
}
