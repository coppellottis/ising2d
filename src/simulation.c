#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"
#include "lattice.h"
#include "metropolis.h"
#include "wolff.h"
#include "observables.h"
#include "io.h"
#include "progress.h"
#ifdef _OPENMP
#include <omp.h>
#endif


// pre-computation of probabilites for Metropolis test
// J is set to 1.
void pre_prob(double p[], double beta) {
    p[0] = exp(-4*beta);
    p[1] = exp(-8*beta);
    return;
}

// --- Thermalization length -------------------------------------------------
//
// This used to be a fixed sweep count (1e4 for Metropolis, 1e3 for Wolff)
// regardless of L. That is very likely a real contributor to the
// "imprecise" autocorrelation times: local Metropolis dynamics have a
// *critical* relaxation time that grows as L^z' (this project's own
// dynamic-z fit gives z' = 2.17 +/- 0.03), so a sweep count that safely
// equilibrates a small lattice can leave a large L run near beta_c far
// from equilibrium. Every observable measured afterwards -- energy,
// magnetization, and especially the autocorrelation time estimate itself
// -- inherits that bias, and it gets worse exactly where L is largest and
// beta is closest to beta_c, i.e. exactly the runs that matter most for
// finite-size scaling.
//
// Wolff's integrated autocorrelation time grows only logarithmically with
// L (Baillie-Coddington), so it needs comparatively little extra
// thermalization even at large L.
//
// These are heuristics (tuned to comfortably exceed L^z' resp. log(L)
// growth with margin), not a rigorous equilibration test -- but they
// track the measured exponents far better than a constant does. Adjust
// the constants below if you profile a specific run.
#define THERM_MIN_METROPOLIS 10000
#define THERM_MULT_METROPOLIS 20.0
#define THERM_Z_METROPOLIS 2.2 // measured z' = 2.17(3), plus a little margin

#define THERM_MIN_WOLFF 1000
#define THERM_MULT_WOLFF 50.0

static int metropolis_therm_sweeps(int L) {
    double scaled = THERM_MULT_METROPOLIS * pow((double)L, THERM_Z_METROPOLIS);
    double n = scaled > THERM_MIN_METROPOLIS ? scaled : THERM_MIN_METROPOLIS;
    return (int)n;
}

static int wolff_therm_sweeps(int L) {
    double scaled = THERM_MULT_WOLFF * log((double)L);
    double n = scaled > THERM_MIN_WOLFF ? scaled : THERM_MIN_WOLFF;
    return (int)n;
}
// ----------------------------------------------------------------------------

void simulation(Lattice* lattice, double beta, const char* alg, const int n_measures, const char* sim_name, pcg32_random_t* rng) {

    FILE* file = open_measurement_file(sim_name, alg, lattice->L, beta);

    // The bar's row is the OpenMP thread slot this job landed on
    // (main.c reserves exactly as many rows as threads it uses).
#ifdef _OPENMP
    int row = omp_get_thread_num();
#else
    int row = 0;
#endif
    char label[48];
    snprintf(label, sizeof(label), "L=%-4d beta=%.4f [%s]", lattice->L, beta, alg);

    if(strcmp(alg, "metropolis") == 0) {
        int N_therm = metropolis_therm_sweeps(lattice->L);

        double* p = malloc(2*sizeof(*p));
        pre_prob(p, beta);

        // Update at most ~100 times per phase: the bar only needs to look
        // smooth, and locking+printf on every single sweep would be
        // wasted overhead over what can be hundreds of thousands of
        // thermalization sweeps. Each phase times itself from its own
        // start (progress_now(), a monotonic clock) so the ETA reflects
        // that phase's actual observed rate, not a guess.
        double therm_t0 = progress_now();
        int therm_step = N_therm/100 > 0 ? N_therm/100 : 1;
        for(int i = 0; i < N_therm; i++) {
            metropolis_sweep(lattice, p, rng);
            if(i % therm_step == 0 || i == N_therm-1) {
                progress_update(row, label, "therm", i+1, N_therm, progress_now()-therm_t0);
            }
        }

        double meas_t0 = progress_now();
        int meas_step = n_measures/100 > 0 ? n_measures/100 : 1;
        for(int i = 0; i < n_measures; i++) {
            metropolis_sweep(lattice, p, rng);
            double E = get_energy(lattice);
            double m = get_magnetization(lattice);
            fprintf(file, "%d,%.10f,%.10f\n", i, E, m);
            if(i % meas_step == 0 || i == n_measures-1) {
                progress_update(row, label, "meas", i+1, n_measures, progress_now()-meas_t0);
            }
        }

        free(p);
    } else if(strcmp(alg, "wolff") == 0) {
        int N_therm = wolff_therm_sweeps(lattice->L);

        double p_add = 1-exp(-2*beta);
        double sweep_frac = 0.0;
        WolffBuffers* buf = wolff_buffers_init(lattice->L);

        double therm_t0 = progress_now();
        int therm_step = N_therm/100 > 0 ? N_therm/100 : 1;
        for(int i = 0; i< N_therm; i++) {
            sweep_frac = wolff_sweep(lattice, p_add, sweep_frac, buf, rng);
            if(i % therm_step == 0 || i == N_therm-1) {
                progress_update(row, label, "therm", i+1, N_therm, progress_now()-therm_t0);
            }
        }

        double meas_t0 = progress_now();
        int meas_step = n_measures/100 > 0 ? n_measures/100 : 1;
        for(int i = 0; i< n_measures; i++) {
            sweep_frac = wolff_sweep(lattice, p_add, sweep_frac, buf, rng);

            double E = get_energy(lattice);
            double m = get_magnetization(lattice);
            fprintf(file, "%d,%.10f,%.10f\n", i, E, m);
            if(i % meas_step == 0 || i == n_measures-1) {
                progress_update(row, label, "meas", i+1, n_measures, progress_now()-meas_t0);
            }
        }

        wolff_buffers_free(buf);
    }

    fclose(file);

    progress_job_done();

    return;
}
