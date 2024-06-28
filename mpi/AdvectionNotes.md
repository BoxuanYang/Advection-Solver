# COMP4300/8300 Parallel Systems Advection Notes, 2024

## Disclaimer

The assignment is designed to achieve a proper balance among solving a problem which to some extent has practical interest (e.g., most numerical weather prediction dynamical cores are based on similar,  but much more complex equations, to simulate atmospheric dynamics) but on the other hand is not too much founded in numerical mathematics up to an extent that you are not able to pursue it if you lack the mathematical background. 

In other words, the assignment is more about computational patterns (e.g., data array layouts and memory access patterns, algorithmic dynamics, etc.) and how to parallelize these using data partitioning (i.e., grid partitioning) and message-passing, rather than requiring a full grasp of the underlying mathematics. To this end, we provide a full serial code in [`serAdvect.c`](https://gitlab.cecs.anu.edu.au/comp4300/2024/comp4300-assignment1/-/blob/master/serAdvect.c?ref_type=heads) to help you out in understanding the aforementioned patterns, no matter the mathematical foundations are grasped or not. 

In real life, research teams are interdisciplinary, and you may find situations where you have to parallelize algorithms without having a full grasp of the foundations of the application domain. 

As we saw in the "Synchronous Computations" lecture, stencil computations arise from applying finite difference discretization methods to the solution of Partial Differential Equations (PDEs).  Most large-scale scientific applications are based on solvers for such equations. 

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

## Some notes about the codes provided

The provided test programs will simulate the advection (motion) of a sine wave across the unit square. An array `u` is set to the field values (i.e. water height) across the square accordingly. The process is iterated over a number of timesteps, and the solution will be the field values in `u` at that point. The boundary conditions are *wrap around* (also known as *periodic*, see above), that is field values at `x = 0` become those at `x = 1` (and conversely). This similarly occurs for the y-dimension. It is possible to compute an exact analytical solution to the advection problem. This can be used to calculate the discretization error in the solution.

The discretization error increases with time (the number of repetitions `r`) and decreases with increasing grid size `(M, N)`. Parallelization of the algorithm, and all other optimizations that we will perform, should produce a numerically identical solution, i.e. have exactly the same discretization error. In our context, this is a very useful property as an increase in the reported error, even a very small one, from a parallelized/optimized version indicates an algorithmic error. This could result from, for example, an error in handling the boundaries.

The boundary conditions are handled as follows. If the size of the field is `MxN`, the array `u` is size `(M+2)x(N+2)`, in order to store an extra row on the top and bottom, and an extra column to the left and right. These are used to store the boundary values (these are also known as *ghost cells*). That is, the corner elements of the halo are at indices *(0,0)*, *(0,N+1)*, *(M+1,N+1)*, *(M+1,0)*; whereas the corner elements of the interior field are at indices *(1,1)*, *(1,N)*, *(M,N)*, *(M,1)*. The elements of the boundary of the inner field are used to form the halo; these points are referred to as the *inner halo*. Due to the (outer) halo, all interior field elements can be updated in a uniform way. In a parallel implementation on a `P` by `Q` process grid, halos are also used to store the interior field elements of the neighbouring processes, for the same reason. Unlike the heat flow problem, the corner points for the halos are used in the update of the corner elements of the interior field. However, by using a 2-stage halo exchange (i.e. top-bottom then left-right), with the 2nd exchange  being of size `M+2` instead of `M`, the corner points can be exchanged implicitly.
