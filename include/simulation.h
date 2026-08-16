#ifndef SIMULATION_H
    #define SIMULATION_H 1

    #include <lattice.h>

    void pre_prob(double* p, double beta);
    void simulation(Lattice* lattice, double beta, const char* alg, const int n_measures, const char* sim_name);
#endif