#ifndef METROPOLIS_H
    #define METROPOLIS_H 1

    #include "lattice.h"
    #include <stdbool.h>

    void metropolis_update(Lattice* lattice, double* p, int* ax);
    void metropolis_sweep(Lattice* lattice, double* p, int* ax);
#endif