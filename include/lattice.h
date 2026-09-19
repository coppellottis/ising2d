#ifndef LATTICE_H
    #define LATTICE_H 1

    #include <stdbool.h>
    #include <stdint.h>
    #include "pcg_basic.h"

    typedef struct{
        int8_t* spins;
        int L;
    } Lattice;

    // rng: the caller's already-seeded, per-job RNG stream. Passing it in
    // (instead of using the global pcg32_random()) keeps the hot start
    // reproducible from the job's seed and safe under OpenMP, where many
    // lattices are initialized concurrently by different threads.
    Lattice* init_lattice(int L, bool hot, pcg32_random_t* rng);
    void free_lattice(Lattice* lattice);
#endif
