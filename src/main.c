#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "pcg_basic.h"
#include "lattice.h"
#include "observables.h"
#include "simulation.h"
#include "metropolis.h"



int main(void){

    double beta = 0.35;

    for(int i=0; i<30; i++) {
        int ax = 0;

        // initializing rng
        uint64_t seed = (uint64_t)time(NULL);
        uint64_t stream = 54u;
        pcg32_srandom(seed, stream);

        // initializing lattice
        Lattice* lat;
        lat = init_lattice(30,true);
        simulation(lat, beta, &ax);
        free_lattice(lat);

        double ax_rate = (double)ax/((1e4+1e5)*(30*30));
        printf("Acceptance rate (L=30, beta=%.4f): %.6f\n", beta, ax_rate);

        beta = beta + (0.2/30.0);
    }
    return 0;
}