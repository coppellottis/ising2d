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

int main(void){

    int n_L;
    char alg[32];
    char sim_name[256];

    printf("Chose a folder name for the new simulation: ");
    scanf("%255s", sim_name);

    char folder_path[512];

    snprintf(folder_path, sizeof(folder_path), "data/%s", sim_name);

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

    for(int i=0; i<n_L; i++) {

        while(1) {
            printf("\n--- Lattice %d ---\n",i+1);
            printf("L = ");
            scanf("%d", L+i);

            if(L>0) break;
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

    for(int i = 0; i < n_L; i++) {

        // SIMULATION

        #pragma omp parallel for
        for(int j = 0; j < n_beta[i]; j++) {
            double beta;
        
            if(n_beta[i] == 1) beta = beta_i[i];
            else beta = beta_i[i] + j*(beta_f[i] - beta_i[i]) / (n_beta[i]-1);

            printf("\nL = %d, beta = %.4f, measurements = %d, algorithm = %s\n", L[i], beta, n_measures[i], alg);

            // initializing rng
            pcg32_random_t rng;

            uint64_t seed = 123456789u + i;
            uint64_t stream = 54u + i;

            pcg32_srandom_r(&rng, seed, stream);

            // initializing lattice
            Lattice* lat;
            lat = init_lattice(L[i],true);
            simulation(lat, beta, alg, n_measures[i], sim_name, &rng);
            free_lattice(lat);
        }
        
        // METADATA 

        char metadata_path[512];
        snprintf(metadata_path, sizeof(metadata_path), "data/%s/metadata.csv", sim_name);
        FILE *metadata = fopen(metadata_path, "a");

        if (metadata == NULL) {
            perror("WARNING: an error occurred while opening metadata.csv");
            exit(EXIT_FAILURE);
        }

        fseek(metadata, 0, SEEK_END);

        if (ftell(metadata) == 0) {
            fprintf(metadata,"algorithm,L,beta_i,beta_f,n_beta,n_measures\n");
        }

        fprintf(metadata,"%s,%d,%f,%f,%d,%d\n",alg,L[i],beta_i[i],beta_f[i],n_beta[i],n_measures[i]);
        fclose(metadata);
    }

    free(L);
    free(beta_i);
    free(beta_f);
    free(n_beta);
    free(n_measures);

    return 0;
}