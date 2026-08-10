#ifndef WOLF_H 
    #define WOLF_H 1

    #include "lattice.h"

    void wolff_update(Lattice* lattice);
    void wolff_sweep(Lattice* lattice);

#endif