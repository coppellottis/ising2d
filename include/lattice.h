#ifndef LATTICE_H
    #define LATTICE_H 1

    #include <stdbool.h>

    typedef struct{
        int* spins;
        int N;
    } Lattice;

    Lattice* init_lattice(int N, bool hot);
    void free_lattice(Lattice* lattice);
#endif