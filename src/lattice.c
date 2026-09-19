#include <stdlib.h>
#include <stdint.h>
#include "lattice.h"
#include "pcg_basic.h"

// Initializes the lattice with random spin if hot=1,
// spin=+1 otherwise.
//
// NOTE (bugfix): this used to draw from the global, non-reentrant
// pcg32_random() instead of the caller's seeded rng. Under
// "#pragma omp parallel for" in main.c that meant every thread hammered
// the same shared PCG state concurrently with no synchronization -- a
// data race (undefined behavior) that also made the hot start ignore the
// per-job seed entirely, so runs weren't actually reproducible. Now it
// uses the thread-local rng passed in from main.c, exactly like every
// other RNG draw in the simulation.
Lattice* init_lattice(int L, bool hot, pcg32_random_t* rng) {
    Lattice* lattice = malloc(sizeof(Lattice));
    lattice->spins = malloc(L*L*sizeof(int8_t));
    lattice->L = L;

    if(hot) {
        for(int i=0; i<(L*L); i++) {
            lattice->spins[i] = pcg32_boundedrand_r(rng, 2) ? 1 : -1;
        }
    } else {
        for(int i=0; i<(L*L); i++) {
            lattice->spins[i] = 1;
        }
    }

    return lattice;
}


void free_lattice(Lattice* lattice) {
    free(lattice->spins);
    free (lattice);
    return;
}
