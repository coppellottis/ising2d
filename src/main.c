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

    // initializing rng
    uint64_t seed = (uint64_t)time(NULL);
    uint64_t stream = 54u;
    pcg32_srandom(seed, stream);

    // initializing lattice
    Lattice* lat;
    lat = init_lattice(30,true);

    // computing main observables
    double m = get_magnetization(lat);
    double E = get_energy(lat);

    // output
    printf("Lattice succesfully initalized!\n");
    printf("The net magnetization m is: %.6f\n", m);
    printf("The energy of the system E is: %.6f J\n", E);

    double beta = .40;
    double* p = malloc(2*sizeof(*p));
    pre_prob(p, beta);

    metropolis_sweep(lat, p);

    m = get_magnetization(lat);
    E = get_energy(lat);

    printf("Lattice succesfully sweeped!\n");
    printf("The net magnetization m is now: %.6f\n", m);
    printf("The energy of the system E is now: %.6f J\n", E);

    // cleaning memory
    free_lattice(lat);
    free(p);

    return 0;
}