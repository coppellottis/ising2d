#include "metropolis.h"
#include "observables.h"
#include "pcg_basic.h"
#include <time.h>
#include <math.h>
#include <stdio.h>

void metropolis_update(Lattice* lattice, double* p) {
    int L = lattice->L;

    // One site r is chosen randomly
    int r = pcg32_boundedrand(L*L);

    // Local field for site r
    int S = get_localfield(lattice, r);

    if(lattice->spins[r]*S <= 0) { 
        // s_r*S means that the local field points in the
        // opposite direction with respect to s_r:
        // the spin flip is accepted since it lowers the energy
        lattice->spins[r] = -lattice->spins[r];
    } else {
        double t = (double)pcg32_random() / 4294967296.0;
        if(t <= p[(int)(lattice->spins[r]*S)/2-1]) {
            lattice->spins[r] = -lattice->spins[r];
        }
    }

    return;
}

void metropolis_sweep(Lattice* lattice, double* p) {
    int L = lattice->L;
    for(int i=0; i < L*L; i++) {
        metropolis_update(lattice, p);
    }
    return;
}