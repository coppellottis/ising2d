#ifndef OBSERVABLES_H
    #define OBSERVABLES_H 1

    #include "lattice.h"

    double get_magnetization(Lattice* lattice);
    double get_energy(Lattice* lattice);
    int get_localfield(Lattice *lattice, int r); 
    void get_nn(Lattice* lattice, int r, int* nn);

#endif