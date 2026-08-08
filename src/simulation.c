#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "simulation.h"
#include "lattice.h"
#include "metropolis.h"
#include "observables.h"

// pre-computation of probabilites for Metropolis test
// J is set to 1.
void pre_prob(double p[], double beta) {
    p[0] = exp(-4*beta);
    p[1] = exp(-8*beta);
    return;
}

void simulation(Lattice* lattice, double beta) {
    int N_therm = 1e4;
    int N_measures = 1e5;

    char filename[256];

    snprintf(filename, sizeof(filename),"results/results_beta_%.4f.csv", beta);
    FILE* file = fopen(filename, "w");
    fprintf(file, "sweep,E_per_site,m\n");

    double* p = malloc(2*sizeof(*p));
    pre_prob(p, beta);
    
    for(int i = 0; i < N_therm; i++) {
        metropolis_sweep(lattice, p);
    }
    for(int i = 0; i < N_measures; i++) {
        metropolis_sweep(lattice, p);
        double E = get_energy(lattice);
        double m = get_magnetization(lattice);
        fprintf(file, "%d,%.10f,%.10f\n", i, E, m);
    }

    fclose(file);
    free(p);
    return;
}