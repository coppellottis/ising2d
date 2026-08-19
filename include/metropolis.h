#ifndef METROPOLIS_H
    #define METROPOLIS_H 1

    #include "lattice.h"
    #include "pcg_basic.h"

    void metropolis_update(Lattice* lattice, double* p, pcg32_random_t* rng);
    void metropolis_sweep(Lattice* lattice, double* p, pcg32_random_t* rng);
#endif