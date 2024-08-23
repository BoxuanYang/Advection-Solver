# Advection Solver
This project is my personal implementation of the advection solver problem in earth science.

## The advection equation

We are aiming to solve a time-space problem, a.k.a. time-dependent or transient problem. This means that the continuous solution of the PDE varies both in space (the square domain) and time, i.e., at each point in time we have a different distribution of the solution in space. The equations read as: 

find $`u(\mathbf{x},t) : \mathcal{D} \times [0,T] \rightarrow R`$ such that:
$$\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = 0 \ \mathrm{in} \ \mathcal{D} \times (0,T] $$
where: 
 * $`u(\mathbf{x},0)=\sin(4.0 \pi x)  \sin(2.0 \pi y)`$ (initial condition, $`t=0`$)
 * $`u((0,y),t)=u((1,y),t)`$ (top boundary=bottom boundary periodic boundary condition, $`t>0`$)
 * $`u((x,0),t)=u((x,1),t)`$ (left boundary=right boundary periodic boundary condition, $`t>0`$)
 * $`\mathcal{D}`$ is the domain where the equations are posed, in our case just an square.
 * $` [0,T]`$ is the time interval in which you want to solve the equation (e.g., $`T=10`$ seconds) 
 * $`\mathbf{x}=(x,y)`$ are the Cartesian coordinates of a point in $`\mathcal{D}`$
 * $`\nabla u`$ is the gradient of $`u`$ (i.e., a vector-valued field with the partial derivatives in space).
 * $`\mathbf{v}`$ is a given/known (vector-valued) velocity field that describes the dynamics of a fluid which is "transporting" the quantity $`u`$. This is known as advection velocity.
* $`\frac{\partial u}{\partial t}`$ is the partial time derivative of $`u`$. 

This equation models the process of *transport*, e.g. wind in an atmosphere simulation. In practice, the most important uses for advection solvers are for 3D phenomena. However, 2D solvers can still model some important problems, e.g. water surface wave propagation, and are considerably simpler to implement. 

## Discretization of the advection equation

To discretize these equations, there are a myriad of numerical methods around. There is indeed still state-of-the-art research being made on novel/improved numerical methods for these equations. In any case, in the assignment we are using a very simple (the most simple) discretization approach. Namely, forward finite differences for the time derivative (also known as Forward Euler time integration method), and an appropriate finite difference formula for the derivatives in space (i.e., for the gradient). The interval [0,T] is split into equal segments of size $`\Delta t`$, and the space domain $`\mathcal{D}`$ is split into a regular space grid of points as we did in the with the Heat Equation in the "Synchronous Computations" lectures. The stencils of the time-space discretization are grounded on the particular finite-difference formulas used for the approximation of the time and space partial derivatives. When solved on a 2D regular Cartesian grid, advection uses a 9-point stencil (unlike the heat flow problem, which is a 5-point stencil). 

Now, Forward Euler is an explicit method, which essentially means that in order to obtain the solution at the next time step from the one at the previous step, we just need to perform matrix-vector products. We do **NOT** have to solve linear systems at each iteration, this is why here there is actually no Jacobi iteration as with the Heat Equation. While computationally cheap, Forward Euler is not unconditionally stable, meaning that as you refine the spatial mesh the time discretization mesh resolution should be adjusted (increased) appropriately according to the so-called [CFL](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) condition to obtain physically meaningful results. 

## Setup



The usage for the test program is:

```bash
OMP_NUM_THREADS=p ./testAdvect [-P P] [-x] M N [r]
```

The test program operates much like that for Assignment 1 except as follows. The `-P` option invokes an optimization where the parallel region is over all timesteps of the simulation, and *P* by *Q* block distribution is used to parallelize the threads, where *p=PQ*. The `-x` option is used to invoke an optional extra optimization.

The directory cuda is similar, except the test program is called `testAdvect.cu`, and the template CUDA parallel solver file is called `parAdvect.cu`. The usage for the test program is:

```bash
./testAdvect [-h] [-s] [-g Gx[,Gy]] [-b Bx[,By]] [-o] [-w w] [-d d] M N [r]
```

with default values of `Gx=Gy=Bx=By=r=1` and `v=w=d=0`. `Gx,Gy` specifies the grid dimensions of the GPU kernels; `Bx,By` specifies the block dimensions.

The option `-h` runs the solver on the host; this may be useful for comparing the 'error' of the GPU runs (and for comparing GPU and CPU speeds). The option `-s` forces a serial implementation (run on a single GPU thread); all other options are ignored. If neither of `-h,-s,-o` are given, `Gx,Gy` thread blocks of size `Bx,By` are used in a 2D GPU parallelization of the solver. If `-o` is specified, an optimized GPU implementation is used, which may use the tuning parameter w as well.

The option `-d` can be used to specify the id of the GPU to be used (`stugpu2` has 4 GPUs, so you can use `d` equal to either `0`, `1`, `2`, or `3`). This may be useful if a particular GPU (e.g. GPU 0) is currently loaded.

You should read the files and familiarize yourself with what they contain.

## Part 1: OpenMP

Experimentation for this section should be done on the Gadi supercomputer. Results should use a batch job run on a single supercomputer node; see the provided file `batchAdv1Node.sh`. Note that for *p ≤ 24*, it forces all threads on one socket (and allocates memory adjacent to the same socket) - this seems to give the best and most consistent results.

Tasks 1. and 2. below are mandatory. Tasks 3. and 4. are optional, and do not contribute to the marks of the assignment. They won't be marked, and as such, feedback won't be provided on these if they are submitted. 
You can use these to test your knowledge in preparation for the final exam.  


1. **Parallelization via 1D Decomposition and Simple Directives**

    Parallelize the functions `omp_update_advection_field_1D_decomposition()` and `omp_copy_field_1D_decomposition()` in `parAdvect.c`. Each loop nesting must have its own parallel region, and the parallelization must be over one of the two loop indices (hint: this can be done via simple OMP parallel for directives).

    For `omp_update_advection_field_1D_decomposition()`, there are various ways this can be done: (i) parallelize the outer or inner loop, (ii) interchange the loop order and (iii) schedule the iterations in a block or cyclic fashion. Choose combinations which:
   
      1. maximize performance.
      2. maximize the number of OpenMP parallel region entry/exits (without significantly increasing the number of cache misses during reads/writes).
      3. maximize cache misses involving read operations (without significantly increasing parallel region entry/exits or cache writes).
      4. maximize cache misses involving write operations (without significantly increasing parallel region entry/exits).
  
    This exercise tests your knowledge regarding cache line transfers. It requires that you understand the parallel memory access patterns resulting from each combination, and how these are laid out in the cache lines of each of the cores. 

    The loops in `omp_update_boundary_1D_decomposition()` can potentially be parallelized as well. Experiment with this for the purpose of improving the performance of case (1) above.

    Cut-and-paste the directive / loop-nesting combinations for each case in your report, and discuss why you chose them. Similarly discuss your strategy in parallelizing the boundary update loops, and whether and by how much it improved the performance of case (1). Choosing suitable values for `M`, `N`,  and `r`, record the performance at various `p` and put the results in your report, discussing the differences.

    Now, leave the best performing parallelization combination in your file.

2. **Parallelization via 2D Decomposition and an Extended Parallel Region** 

    In the function `run_parallel_omp_advection_2D_decomposition()`, write an advection solver such there is only a single parallel region (over all timesteps), and the parallelization is over a `P` by `Q` block distribution. *Hint:* each thread could call `update_advection_field()` and `copy_field()` over its respective sub-array; alternately you could 'inline' these functions and restrict the `i,j` loops to operate on the thread's sub-array.

    For suitable measurements with the best-performing `p` from Q1 and varying `P`, compare performance. Discuss whether there is any gain from the 2D distribution. Comparing this with the 1D vs 2D case for the MPI version of the solver, and explain any differences with the relative advantages. Comparing with your results in Q1 for the `P=p` case, was there any advantage in having a single parallel region?






## Part 2: CUDA

Unless otherwise specified, experimental results for this section should be made on `stugpu2.anu.edu.au`, as described in [Lab 7](https://gitlab.cecs.anu.edu.au/comp4300/2024/comp4300-lab6). *Please note that this is a shared resource and choose your parameters so that the advection time is about 1 second or smaller: this should be plenty long enough to demonstrate performance!* Click [here](https://comp.anu.edu.au/courses/comp4300/assignments_workflow/#access-and-usage-of-stugpu2anueduau-gpu-programming-with-cuda) for `stugpu2.anu.edu.au` access instructions. 



5. **Baseline GPU Implementation**

   Using the code of `run_serial_advection_device()` and its kernels in `serAdvect.cu` as a starting point, implement a solver whose field update kernels operate on $`Gx \times Gy`$ thread blocks of size $`Bx \times By`$ (without restrictions, except you may assume $`Bx*By \leq`$ the maximum thread block size). You may choose what, if any, parallelization strategy you apply for the boundary updates (justify this in your report).

   In `parAdvect.cu`, you will need to create new kernel functions, and your outer-level solver calling these must be implemented in `run_parallel_cuda_advection_2D_decomposition()`. *Hint:* to help in debugging, replace the kernels from `serAdvect.cu` one by one with your own, and do the simpler parallelizations first.

    Perform experiments to determine the effects of varying `Gx,Gy,Bx,By` (given some reasonably sized problem to solve). Report and discuss the optimal combination, and any results of interest (including a comparison of $`1 \times B`$ vs $`B \times 1`$ blocks).

    Perform suitable experiments to determine the overhead of a kernel invocation, and report your findings. Include also some experiments to determine speedups against the single GPU and host (x86-64) cores (but do not use much smaller parameters, as a single GPU core is very, very, slow...).

6. **Optimized GPU Implementation**
    
    In `run_parallel_cuda_advection_optimized()`, create an optimized solver and its associated kernels. It should be (in principle) significantly faster than the previous version. You might have ideas of what optimizations you might consider from Assignment 1; if not, read the paper [<span style="color:blue">Optimized Three-Dimensional Stencil Computation on Fermi and Kepler GPUs</span>](Opt3Dstencils.pdf) by Vizitiu et al. Please note that the architecture used in this paper is different from the GPUs you will use on stugpu2 and Gadi. In your report, describe your optimization(s) and their rationale. Perform suitable experiments which should demonstrate the efficacy of your approach, and discuss them in your report.

