#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <omp.h>
#include "pcg_basic.h"
#include "lattice.h"
#include "observables.h"
#include "simulation.h"
#include "metropolis.h"
#include "io.h"
#include "progress.h"

typedef struct {
    int i;
    int j;
} Job;


int main(void){

    int n_L;
    char alg[32];
    char sim_name[256];

    printf("Chose a folder name for the new simulation: ");
    scanf("%255s", sim_name);

    char folder_path[512];

    snprintf(folder_path, sizeof(folder_path), "data/%s", sim_name);

    // "data/" itself is gitignored (it's the output directory), so a fresh
    // clone won't have it yet. Create it first (ignoring the "already
    // exists" case) so a first run doesn't fail with an unhelpful
    // "No such file or directory" before ever getting to the real target
    // folder.
    if (mkdir("data", 0777) != 0 && errno != EEXIST) {
        perror("WARNING: could not create data/ folder.");
        exit(EXIT_FAILURE);
    }

    if (mkdir(folder_path, 0777) != 0) {
        perror("WARNING: could not create folder.");
        exit(EXIT_FAILURE);
    }

    while(1) {
        printf("Choose the algorithm (metropolis/wolff): ");
        scanf("%31s", alg);
        if(strcmp(alg, "metropolis") == 0 || strcmp(alg, "wolff") == 0){
            break;
        }
        printf("WARNING: the selected algorithm is invalid. "
        "Please, use 'metropolis' or 'wolff' to select one.\n");
    }

    while(1) {
        printf("How many lattice sizes do you want to study? ");
        scanf("%d", &n_L);

        if(n_L > 0) break;
        printf("WARNING: the number of lattice sizes must be > 0.\n");
    }

    int* L = calloc(n_L, sizeof(int));
    double* beta_i = calloc(n_L, sizeof(double));
    double* beta_f = calloc(n_L, sizeof(double));
    int* n_beta = calloc(n_L, sizeof(int));
    int* n_measures = calloc(n_L, sizeof(int));
    int n_jobs = 0;

    for(int i=0; i<n_L; i++) {

        while(1) {
            printf("\n--- Lattice %d ---\n",i+1);
            printf("L = ");
            scanf("%d", L+i);

            if(L[i]>0) break;
            printf("WARNING: L must be > 0.\n");
        }

        while(1) {
            printf("Initial beta = ");
            scanf("%lf", beta_i+i); // note for me: %f (input) -> %lf (output) for float

            if(beta_i[i]>0) break;
            printf("WARNING: initial beta must be > 0.\n");
        }      

        while(1) {
            printf("Final beta = ");
            scanf("%lf", beta_f+i); // note for me: %f (input) -> %lf (output) for float

            if(beta_f[i]>= beta_i[i]) break;
            printf("WARNING: final beta must be >= initial beta.\n");
        }       
        
        while(1) {
            if(beta_f[i] == beta_i[i]) {
                n_beta[i] = 1;
            } else {
                printf("Number of beta values = ");
                scanf("%d", n_beta+i);
            }
            n_jobs += n_beta[i];

            if(n_beta[i] > 0) break;
            printf("WARNING: the number of beta values must be > 0.\n");
        }
        
        while(1) {
            double a;
            printf("Number of measurements = ");
            scanf("%lf", &a);
            n_measures[i] = (int) a;
            
            if(n_measures[i] > 0) break;
            printf("WARNING: the number of measurements must be > 0.\n");
        }
    }

    write_metadata(sim_name, alg, L, beta_i, beta_f, n_beta, n_measures, n_L);

    Job *jobs = malloc(n_jobs * sizeof(Job));

    int k = 0;
    for (int i = 0; i < n_L; i++)
        for (int j = 0; j < n_beta[i]; j++)
            jobs[k++] = (Job){i, j};

    // Cap the thread count to the number of jobs (no point reserving idle
    // progress-bar rows for threads that will never run anything), and
    // reserve exactly that many bar rows before entering the parallel
    // region -- progress_init() must run single-threaded, and no other
    // printf should land inside that reserved screen area afterwards or
    // it'll scroll the bars out of sync with the cursor math.
    int nt = omp_get_max_threads();
    if (nt > n_jobs) nt = n_jobs;
    if (nt < 1) nt = 1;

    printf("\nRunning %d job%s on %d thread%s...\n", n_jobs, n_jobs == 1 ? "" : "s", nt, nt == 1 ? "" : "s");
    progress_init(nt, n_jobs);

    // SIMULATION
    #pragma omp parallel for num_threads(nt)
    for(int k = 0; k < n_jobs; k++) {

        int i = jobs[k].i;
        int j = jobs[k].j;

        double beta;

        if(n_beta[i] == 1) beta = beta_i[i];
        else beta = beta_i[i] + j*(beta_f[i] - beta_i[i]) / (n_beta[i]-1);

        // initializing rng
        pcg32_random_t rng;

        // Bugfix: seeding used to be "123456789u + i+j*1000" / "54u + i+j*1000".
        // For any scan with n_beta[i] >= ~1000 (or enough lattice sizes),
        // different (i,j) pairs could map to the same i+j*1000 and silently
        // reuse the exact same seed AND stream for two different jobs. The
        // flat job index k is unique across the whole batch by construction
        // (see the "jobs" array above), so deriving seed/stream from it can't
        // collide regardless of how many L's or beta's are scanned.
        uint64_t seed = 123456789u + (uint64_t)k;
        uint64_t stream = 987654321u + (uint64_t)k;

        pcg32_srandom_r(&rng, seed, stream);

        // initializing lattice
        Lattice* lat;
        lat = init_lattice(L[i], true, &rng);
        simulation(lat, beta, alg, n_measures[i], sim_name, &rng);
        free_lattice(lat);
    }

    progress_finish();
    printf("Done: %d job%s completed.\n", n_jobs, n_jobs == 1 ? "" : "s");

    free(jobs);
    free(L);
    free(beta_i);
    free(beta_f);
    free(n_beta);
    free(n_measures);

    return 0;
}