# COMP4300/8300 Parallel Systems Assignment 1, 2024
# A Distributed Advection Solver Using MPI

## Deadline: 15/04/2024, 11:55PM

*(Please report any errors, omissions and ambiguities to the course lecturers)* \
*(We may make small amendments and fixes to this document after its release; please keep an eye on the commit history after release date just in case)*

This assignment is worth 25% of your total course mark. It will be marked out of 50.

## Assignment Setup and Submission

This assignment must be submitted electronically. To obtain a copy of the template source files, you must fork the gitlab project [`comp4300-assignment1`](https://gitlab.cecs.anu.edu.au/comp4300/2024/comp4300-assignment1) into your own gitlab area, and push the final version of your files into your fork of the `comp4300-assignment1` gitlab project before the deadline.

The files that will be marked are a report `ps-ass1Rep.pdf` and a C source code file `parAdvect.c`. The report must be **at most** 2,000 words long. A good report will present information in an easy to digest format - for example figures and plots are often much easier to interpret than raw tables of data. You should avoid putting snippets of code in the report and instead keep the discussion at a high level, referring the reader to the appropriate locations in your code where appropriate. **The report has a large relative weight in the final mark, and should be well-written. Submissions without report will get a very low mark.** 

Please see [here](https://comp.anu.edu.au/courses/comp4300/assignments_workflow/) for additional workflow notes. Most importantly, use the login node only for debugging purposes on a moderate number of MPI tasks (e.g., 1-8). The actual performance analysis should be done on the compute nodes using the job scripts provided in this repo. 


## Learning Objectives

- Implement a distributed algorithm using message passing.
- Model the computation and communication costs of the algorithm.
- Measure the performance of the implemented code, and compare against the model.
- Research and evaluate relevant parallel computing literature. 

See the [Advection Notes](./AdvectionNotes.md) for the necessary background. 

## Setup

Your assignment project directory will contain a test program `testAdvect.c`, a file `serAdvect.c` containing a serial advection solver and support functions, some header files, and a template parallel advection solver `parAdvect.c`. The test program can be built using the command `make`. 

It also contains a template `ps-ass1Rep.pdf`, which you must overwrite with your own report. 

The usage for the test program is: 

```bash
mpirun -np p ./testAdvect [-P P] [-w w] [-o] [-x] M N [r] 
```

with default values `P=p, w=1, r=1`. This will run an `M` by `N` advection simulation over `r` timesteps (repetitions) using a `P` by `Q` process grid, where `p=PQ`. `w` specifies the halo width (normally it is 1). If `-o` is specified, halo communication should be overlapped with computation. The `-x` is used to invoke an optional extra optimization. 

There is also a `-v` option which can be used for debugging (try using `-v 1`, `-v 2` etc). 

The test program initializes the local field array `u` with leading dimension `ldu`, calls the appropriate parallel advection function (in `parAdvect.c`), and determines the error in the final field. It assumes a 2D block distribution of the global advection field. However, `parAdvect.c` determines the details of this distribution (i.e. how to deal with remainders), and exports: 

* `M_loc`, `N_loc`: the local advection field size (excluding the halo). 
* `M0`, `N0`: the local field element `(0,0)` is global element `(M0,N0)` 

You should read the files and familiarize yourself with what they contain. 

## Your Tasks 

For multi-node runs, experimentation for this section should be done on the Gadi supercomputer using batch jobs. Recall from Lab 1 that Gadi time allocations are measured in *Service Units* (SU). For this course, we have a budget of 250SU per student. *This will not refresh, and if you exceed this limit then you will no longer be able to submit jobs*. It is of the utmost importance that you manage your allocation carefully. 

1. **Deadlock issues** (Mark: 3/50)

The prototype function `run_parallel_advection()` in `parAdvect.c` should work for `Q=1`. However, it can deadlock (try `p=2`, `M=2` and a large `N`). In your report, write down for what values of `N` it deadlocks, and explain why. Without using non-blocking send or receives, or buffered sends, fix the halo-exchange code in `parAdvect.c` to avoid the deadlock regardless of program parameters. Describe the fixed code in your report. 

2. **The effect of non-blocking communication** (Mark: 3/50)

Rewrite your halo-exchange code to use non-blocking sends and receives. Use CPP preprocessing directives so you can activate either version of communication. Do some experiments on up to 4 nodes to determine if this improves performance (choose parameters to maximize the impact of communication), and write the results in your report. 

3. **Performance modelling and calibration** (Mark: 8/50)

In your report, write a performance model for the computation, in terms of the above program parameters, and the coefficients for communication startup time ( $`t_s`$ ), communication cost per word time ( $`t_w`$ ), and per element computation time ( $`t_f`$ ) for the advection solver. 

Run experiments to determine the values of these parameters on Gadi, and justify your methodology. *Hint:* you can use the test program itself with suitably chosen parameters to determine $`t_f`$; we have measured the communication costs previously (see Lab 1 for more details). Comment whether the values are (roughly) what you would expect for Gadi. 

Within one Gadi node, perform a strong scaling analysis and compare predicted vs actual execution time for various numbers of processes `p`. Use the same value of `M`, `N`, and `r` throughout. Justify why you think those values are suitable. *Hint:* `r=100` should be big enough to amortize startup effects; we suggest keeping the field size to fit within shared L3 cache of a single socket of the Intel Xeon 8274 (Cascade Lake). 

Present the results in your report, commenting on any discrepancies. *Hint:* at certain points, other parts of the memory hierarchy may have an effect on the timings. 

4. **The effect of 2D process grids** (Mark: 8/50)

Extend your code in `run_parallel_advection()` for two-dimensional process grids, i.e. `Q>1`. Extend your performance model accordingly. 

*Hints:* the existing code assumes `Q=1`. You will need to modify `init_parallel_parameter_values()` for `Q>1`, possibly some of the top-bottom halo exchange in `update_boundary()` as well as adding message passing code for the left-right exchange. 2D block distributions have a number of valid variants; you may choose any of these. 

Conduct experiments with fixed `M=N` on one node to see if the process grid aspect ratio has any effect on performance. Of most interest is whether a (near) square ratio has a different effect to the default (`Q=1`). What is the optimum aspect ratio predicted by your model? Use your model to predict how much difference there would be if the coefficient `tw` was 10 times larger. 

Also conduct experiments on 1 to 4 nodes (48 to 192 cores). Do this also for experiments in the questions below. *Hint:* restrict the problem size to fit within the combined L3 cache of 2 Cascade Lake sockets (48 cores). 

5. **Overlapping communication with computation** (Mark: 8/50)

An advanced technique in message-passing parallel algorithms is to hide communication costs by updating the data that must be communicated (for the next step) first, and send this before updating the rest of the data. In this case, it is the inner halo. Implement this technique in `run_parallel_advection_with_comp_comm_overlap()`. 

In your report, discuss what the performance impact of this technique might be, and describe how it would affect your performance model. Using the `-o` option, run appropriate experiments to determine whether it is effective, and record and discuss in your report. It will be sufficient to implement this just for 1D process grids, i.e. for `Q=1`. However, in this case, you should explain why achieving overlap for 2D communication is difficult. 

6. **Wide halo transfers** (Mark: 8/50)

In stencil computations, wide halos are a technique intended to reduce parallel overheads. Instead of exchanging a halo of width 1 at process boundaries every simulated timestep, a halo of width `w >= 1` can be exchanged every `w` timesteps. In each process, let the local advection field size (without halos) be `(m)x(n)`. You will notice that the test program allocates an `(m+2*w)x(n+2*w)` array to support this. Using the widened halo, we perform `w` updates to the local field of sizes `(m+2*w-2)x(n+2*w-2), (m+2*w-4)x(n+2*w-4), ..., (m)x(n)`.

Like the normal halo, the widened halos are generated from the interior points on the current or neighbouring processes. However this is problematic if `w > m, n`. The function `check_halo_size()` checks for this problem, but creates an untidy shutdown. Fix this so that if the above condition is violated in any process, all processes exit and only a single error message is printed out. 

In your report, discuss what you think are the potential advantages of wide halos, and describe how it would affect your performance models. In the function `run_parallel_advection_with_wide_halos()` in `parAdvect.c`, extend your 2D code with the wide halo technique and test it using the `-w` option. Run appropriate experiments to determine whether it is effective, and record and discuss in your report. 

7. **Literature Review (optimization techniques for stencil computations)** (Mark: 6/50)

Search the literature and find different optimization techniques in order to improve the performance of stencil computations. (*Hint:* the tiled stencil technique may be interesting.)  Report on two selected ones among those which you found.  Provide a summary for each including its motivation and effectiveness (150-400 words). Fully cite all your sources (2+ sources are expected). 

8. **Performance outcome via combination of optimization techniques** (Mark: 2/50)

The performance of stencil computations is primarily limited by memory system performance. Briefly explain whether a combination of two optimization techniques from the ones that you found in the point above may alleviate this, and if so what are the performance trade-offs of such an approach.

9. **Implement an optimization technique** (Mark: 4/50)

Implement an optimization to improve the performance of the parallel advection solver. This might be different from the ones discussed in question 7, but not necessarily. Evaluate the performance of the optimization technique on Gadi. Put your code in `run_parallel_advection_with_extra_opts()`, which is activated by the `-x` option. 

## Requirements

Your code in `parAdvect.c` will be tested against the standard `serAdvect.c` and `testAdvect.c` and must compile and run correctly with them. Solutions which are highly incompatible may be rejected entirely. Your code should be written with good programming style (and so should not produce any warnings when compiled with the `-Wall` option!). It should be well commented, so as to enable the reader (this includes the tutor/marker!) to understand how it works. 

In your report, include: 

* A disclaimer with your name and student number stating what (if any) contributions from others (not including course staff) are in your submission. Note that significant contributions may require a revision of your final mark, as this is intended to be an individual assignment. 
* Your answers to all the questions given above. 
* Details of any notable deficiencies, bugs, or issues with your work. 
* Any feedback on what you found helpful in doing this work, or what from the course could be improved in helping you to carry out this work. 
