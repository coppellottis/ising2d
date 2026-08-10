#ifndef METROPOLIS_H
    #define METROPOLIS_H 1

    #include "lattice.h"

    void metropolis_update(Lattice* lattice, double* p);
    void metropolis_sweep(Lattice* lattice, double* p);
#endif