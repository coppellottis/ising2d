#ifndef LATTICE_H
    #define LATTICE_H 1

    #include <stdbool.h>

    typedef struct{
        int* spins;
        int L;
    } Lattice;

    Lattice* init_lattice(int L, bool hot);
    void free_lattice(Lattice* lattice);
#endif