#include <stdio.h>
#include <stdlib.h>
#include "io.h"

void write_metadata(const char* sim_name, const char* alg, const int* L,
                     const double* beta_i, const double* beta_f,
                     const int* n_beta, const int* n_measures, int n_L) {
    char metadata_path[512];
    snprintf(metadata_path, sizeof(metadata_path), "data/%s/metadata.csv", sim_name);

    FILE* metadata = fopen(metadata_path, "w");
    if (metadata == NULL) {
        perror("WARNING: could not open metadata.csv for writing");
        exit(EXIT_FAILURE);
    }

    fprintf(metadata, "algorithm,L,beta_i,beta_f,n_beta,n_measures\n");
    for (int i = 0; i < n_L; i++) {
        fprintf(metadata, "%s,%d,%f,%f,%d,%d\n", alg, L[i], beta_i[i], beta_f[i], n_beta[i], n_measures[i]);
    }

    fclose(metadata);
}

FILE* open_measurement_file(const char* sim_name, const char* alg, int L, double beta) {
    char filename[256];
    snprintf(filename, sizeof(filename), "data/%s/%s_L%d_beta%.4f.csv", sim_name, alg, L, beta);

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("WARNING: could not open measurement file for writing");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "sweep,E_per_site,m\n");
    return file;
}
