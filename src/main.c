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

    for(int i=0; i<n_L; i++) {
        int L;
        double beta_i, beta_f;
        int n_beta;
        int n_measures;

        while(1) {
            printf("\n--- Lattice %d ---\n",i+1);
            printf("L = ");
            scanf("%d", &L);

            if(L>0) break;
            printf("WARNING: L must be > 0.\n");
        }

        while(1) {
            printf("Initial beta = ");
            scanf("%lf", &beta_i); // note for me: %f (input) -> %lf (output) for float

            if(beta_i>0) break;
            printf("WARNING: initial beta must be > 0.\n");
        }      

        while(1) {
            printf("Final beta = ");
            scanf("%lf", &beta_f); // note for me: %f (input) -> %lf (output) for float

            if(beta_f>= beta_i) break;
            printf("WARNING: final beta must be >= initial beta.\n");
        }       
        
        while(1) {
            if(beta_f == beta_i) {
                n_beta = 1;
            } else {
                printf("Number of beta values = ");
                scanf("%d", &n_beta);
            }

            if(n_beta > 0) break;
            printf("WARNING: the number of beta values must be > 0.\n");
        }
        
        while(1) {
            double a;
            printf("Number of measurements = ");
            scanf("%lf", &a);
            n_measures = (int) a;
            
            if(n_measures > 0) break;
            printf("WARNING: the number of measurements must be > 0.\n");
        }

        // SIMULATION

        #pragma omp parallel for
        for(int i = 0; i < n_beta; i++) {
            double beta;
        
            if(n_beta == 1) beta = beta_i;
            else beta = beta_i + i*(beta_f - beta_i) / (n_beta-1);

            printf("\nL = %d, beta = %.4f, measurements = %d, algorithm = %s\n", L, beta, n_measures, alg);

            // initializing rng
            pcg32_random_t rng;

            uint64_t seed = 123456789u + i;
            uint64_t stream = 54u + i;

            pcg32_srandom_r(&rng, seed, stream);

            // initializing lattice
            Lattice* lat;
            lat = init_lattice(L,true);
            simulation(lat, beta, alg, n_measures, sim_name, &rng);
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

        fprintf(metadata,"%s,%d,%f,%f,%d,%d\n",alg,L,beta_i,beta_f,n_beta,n_measures);
        fclose(metadata);
    }
    return 0;
}