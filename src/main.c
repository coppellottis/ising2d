#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "pcg_basic.h"
#include "lattice.h"
#include "observables.h"
#include "simulation.h"
#include "metropolis.h"



int main(void){

    double beta_i = 0.42;
    double beta_f = 0.46;
    int n_champ = 20;

    for(int i=0; i<n_champ; i++) {
        double beta = beta_i + i*((beta_f-beta_i)/n_champ);
        for(int j=0; j<5; j++) {

            int L = (int)pow(2,j)*8;

            // initializing rng
            uint64_t seed = (uint64_t)time(NULL);
            uint64_t stream = 54u;
            pcg32_srandom(seed, stream);

            // initializing lattice
            Lattice* lat;
            lat = init_lattice(L,true);
            simulation(lat, beta);
            free_lattice(lat);
        }
    }
    return 0;
}