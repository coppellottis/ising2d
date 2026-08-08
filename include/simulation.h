#ifndef SIMULATION_H
    #define SIMULATION_H 1

    #include <lattice.h>

    void pre_prob(double* p, double beta);
    void simulation(Lattice* lattice, double beta, int* ax);
#endif