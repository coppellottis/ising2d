#ifndef WOLF_H 
    #define WOLF_H 1

    #include "lattice.h"

    int wolff_update(Lattice* lattice, double p_add);
    double wolff_sweep(Lattice* lattice, double p_add, double sweep_frac);

#endif