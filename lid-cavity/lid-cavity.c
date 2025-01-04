#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Apply the boundary conditions to the u and v arrays.
 * 
 */
void apply_boundary() {
	for (int i = 0; i < nx; i++) {
		u[i][0] = 0;
		u[i][ny-1] = 0;
		v[i][0] = 0;
		v[i][ny-1] = 0;
	}

	for (int j = 0; j < ny; j++) {
		u[0][j] = 0;
		u[nx-1][j] = 1; // set the velocity on the cavity lid to 1.0
		v[0][j] = 0;
		v[nx-1][j] = 0;
	}
}

/**
 * @brief Build the right hand side of the Poisson equation.
 * 
 */
void build_rhs() {
	// build the "b" matrix (RHS)
	for (int i = 1; i < nx-1; i++) {
		for (int j = 1; j < ny-1; j++) {
			b[i][j] = (rho * (1 / dt) *
				((u[i][j+1] - u[i][j-1]) /
				(2 * dx) + (v[i+1][j] - v[i-1][j]) / (2 * dy)) -
				pow((u[i][j+1] - u[i][j-1]) / (2 * dx), 2) -
				2 * ((u[i+1][j] - u[i-1][j]) / (2 * dy) *
				(v[i][j+1] - v[i][j-1]) / (2 * dx)) -
				pow((v[i+1][j] - v[i-1][j]) / (2 * dy), 2));
		}
	}
}

/**
 * @brief Solve the Poisson equation to calculate the pressure across the domain.
 * 
 */
void solve_poissons() {
	double **pn = alloc_2d_array(nx,ny);

	int poisson_iters = 0;
	while (poisson_iters < nit) {
		for (int i = 0; i < nx; i++) 
			for (int j = 0; j < ny; j++)
				pn[i][j] = p[i][j];
	
		for (int i = 1; i < nx-1; i++) {
			for (int j = 1; j < ny-1; j++) {
				p[i][j] = (((pn[i][j+1] + pn[i][j-1]) * (dy*dy) +
					(pn[i+1][j] + pn[i-1][j]) * (dx*dx)) /
					(2 * (dx*dx + dy*dy)) -
					(dx*dx) * (dy*dy) / (2 * (dx*dx + dy*dy)) * 
					b[i][j]);
			}
		}
		for (int i = 0; i < nx; i++) {
			p[i][ny-1] = p[i][ny-2]; // dp/dx = 0 at x = 2
			p[i][0] = p[i][1];       // dp/dx = 0 at x = 0   
		}
		for (int j = 0; j < ny; j++) {
			p[0][j] = p[1][j];        // dp/dy = 0 at y = 0
			p[nx-1][j] = 0;           // p = 0 at y = 2
		}
		poisson_iters++;
	}

	free_2d_array(pn);
}

/**
 * @brief Update the u and v velocity arrays using the calculated pressure.
 * 
 */
void update_velocities() {
	// make a copy of the u and v matrices
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			un[i][j] = u[i][j];
			vn[i][j] = v[i][j];
		}
	}

	for (int i = 1; i < nx-1; i++) {
		for (int j = 1; j < ny-1; j++) {
			u[i][j] = (un[i][j]-
				un[i][j] * dt / dx *
				(un[i][j] - un[i][j-1]) -
				vn[i][j] * dt / dy *
				(un[i][j] - un[i-1][j]) -
				dt / (2 * rho * dx) * (p[i][j+1] - p[i][j-1]) +
				nu * (dt / (dx*dx) *
				(un[i][j+1] - 2 * un[i][j] + un[i][j-1]) +
				dt / (dy*dy) *
				(un[i+1][j] - 2 * un[i][j] + un[i-1][j])));

			v[i][j] = (vn[i][j] -
				un[i][j] * dt / dx *
				(vn[i][j] - vn[i][j-1]) -
				vn[i][j] * dt / dy *
				(vn[i][j] - vn[i-1][j]) -
				dt / (2 * rho * dy) * (p[i+1][j] - p[i-1][j]) +
				nu * (dt / (dx*dx) *
				(vn[i][j+1] - 2 * vn[i][j] + vn[i][j-1]) +
				dt / (dy*dy) *
				(vn[i+1][j] - 2 * vn[i][j] + vn[i-1][j])));
		}
	}
}

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	set_defaults();
	parse_args(argc, argv);
	setup();

	printf("Running problem size %f x %f on a %d x %d grid.\n", X, Y, nx, ny);
	
	if (verbose) print_opts();
	
	allocate_arrays();

	x = calloc(nx, sizeof(double));
	for (int i = 0; i < nx; i++) x[i] = (2.0 / (nx-1)) * i;
	y = calloc(ny, sizeof(double));
	for (int i = 0; i < ny; i++) y[i] = (2.0 / (ny-1)) * i;

	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);  // record start time

	int iters = 0;
	double t = 0.0;

	while (iters < n_iters) {
		build_rhs();

		solve_poissons();

		update_velocities();

		apply_boundary();

		if (iters % output_freq == 0) {
			printf("Step %8d, Time: %14.8e (dt: %14.8e)\n", iters, t, dt);

			if ((!no_output) && (enable_checkpoints))
				write_checkpoint(iters);
		}
	
		iters++;
		t += dt;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_sec = (end.tv_sec - start.tv_sec)
                       + (end.tv_nsec - start.tv_nsec)*1e-9;
    printf("Total simulation time: %f seconds\n", elapsed_sec);

	printf("Step %8d, Time: %14.8e (dt: %14.8e)\n", iters, t, dt);
	printf("Simulation complete.\n");

	if (!no_output) 
		write_result();

	free_arrays();

	exit(0);
}

