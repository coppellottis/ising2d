#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"
#include "lattice.h"
#include "metropolis.h"
#include "wolff.h"
#include "observables.h"


// pre-computation of probabilites for Metropolis test
// J is set to 1.
void pre_prob(double p[], double beta) {
    p[0] = exp(-4*beta);
    p[1] = exp(-8*beta);
    return;
}

void simulation(Lattice* lattice, double beta, const char* alg, const int n_measures, const char* sim_name) {
    int N_therm = 1e4; 

    char filename[256];

    snprintf(filename, sizeof(filename),"data/%s/%s_L%d_beta%.4f.csv", sim_name, alg, lattice->L, beta);
    FILE* file = fopen(filename, "w");
    fprintf(file, "sweep,E_per_site,m\n");

    if(strcmp(alg, "metropolis") == 0) {
        double* p = malloc(2*sizeof(*p));
        pre_prob(p, beta);

        for(int i = 0; i < N_therm; i++) {
            metropolis_sweep(lattice, p);
        }
        for(int i = 0; i < n_measures; i++) {
            metropolis_sweep(lattice, p);
            double E = get_energy(lattice);
            double m = get_magnetization(lattice);
            fprintf(file, "%d,%.10f,%.10f\n", i, E, m);

            int progress = (int)(100.0 * (i + 1) / n_measures);
            int bars = progress / 2;

            printf("\r[");

            for (int j = 0; j < 50; j++) {
                if (j < bars)
                    printf("=");
                else
                    printf(" ");
            }
            printf("] %3d%%", progress);
            fflush(stdout); // force output
        }

        free(p);
    } else if(strcmp(alg, "wolff") == 0) {
        double p_add = 1-exp(-2*beta);
        double sweep_frac = 0.0;

        for(int i = 0; i< N_therm; i++) {
            sweep_frac = wolff_sweep(lattice, p_add, sweep_frac);
        }

        for(int i = 0; i< n_measures; i++) {
            sweep_frac = wolff_sweep(lattice, p_add, sweep_frac);

            double E = get_energy(lattice);
            double m = get_magnetization(lattice);
            fprintf(file, "%d,%.10f,%.10f\n", i, E, m);

            int progress = (int)(100.0 * (i + 1) / n_measures);
            int bars = progress / 2;

            printf("\r[");

            for (int j = 0; j < 50; j++) {
                if (j < bars)
                    printf("=");
                else
                    printf(" ");
            }
            printf("] %3d%%", progress);
            fflush(stdout); // force output
        }
    }
    
    fclose(file);
    
    printf("\n");
    return;
}