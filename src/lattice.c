#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "lattice.h"
#include "pcg_basic.h"

// Initializes the lattice with random spin if hot=1,
// spin=+1 otherwise.
Lattice* init_lattice(int L, bool hot) {
    Lattice* lattice = malloc(sizeof(Lattice));
    lattice->spins = malloc(L*L*sizeof(int));
    lattice->L = L;

    if(hot) {
        uint64_t seed = (uint64_t)time(NULL);
        uint64_t stream = 54u;

        // pcg_srandom(seed, stream indipendente)
        pcg32_srandom(seed, stream);

        for(int i=0; i<(L*L); i++) {
            // 4294967296.0 = 2^32
            double r = (double)pcg32_random() / 4294967296.0;
            if(r>.5) lattice->spins[i] = 1;
            else lattice->spins[i] = -1;
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