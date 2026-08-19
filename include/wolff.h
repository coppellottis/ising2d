#ifndef WOLF_H 
    #define WOLF_H 1

    #include "lattice.h"
    #include "pcg_basic.h"

    int wolff_update(Lattice* lattice, double p_add, pcg32_random_t* rng);
    double wolff_sweep(Lattice* lattice, double p_add, double sweep_frac, pcg32_random_t* rng);

#endif