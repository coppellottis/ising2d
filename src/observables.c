#include <stdlib.h>
#include "observables.h"
#include "lattice.h"

double get_magnetization(Lattice* lattice) {
    double M = 0;
    int L = lattice->L;
    for(int i=0; i < L*L; i++) {
        M = M+lattice->spins[i];
    }
    return M/(L*L);
}

// In order to avoid double counting, the sum
// over nearest neighbours is performed by considering only
// the right and the down neighbours. J=1.
double get_energy(Lattice* lattice) {
    double E = 0;
    int L = lattice->L;
    for(int i=0; i<L; i++) {
        for(int j=0; j<L; j++) {
            int x = i*L+j;
            int right = i*L+(j+1)% L;
            int down = ((i+1)%L)*L+j;
            E += -lattice->spins[x]*(lattice->spins[right]+lattice->spins[down]);
        }
    }
    return E/(L*L);
}

// Returns the local field on site r. PBC implemented.
int get_localfield(Lattice* lattice, int r) {
    int nn[4];
    get_nn(lattice, r, nn);

    int S = 0;
    for(int i = 0; i<4; i++) {
        S += lattice->spins[nn[i]];
    }
    return S;
}

// Returns nearest neighbours (PBC).
//
// Perf note: right/left already avoided "%" for the common (non-boundary)
// case; down/up used an unconditional "%" on every single call even
// though wraparound only ever happens for sites on the first/last row.
// This is the hottest function in the whole simulation (called on every
// Metropolis attempt and every Wolff cluster-growth step), so replacing
// the two remaining modulo ops with the same branch-on-boundary pattern
// is a cheap, behavior-preserving win.
 void get_nn(Lattice* lattice, int r, int* nn) {
    int L = lattice->L;
    int L2 = L*L;

    nn[0] = (r % L == L - 1) ? r - L + 1 : r + 1; // right
    nn[1] = (r % L == 0) ? r + L - 1 : r - 1; // left
    nn[2] = (r + L < L2) ? r + L : r + L - L2; // down
    nn[3] = (r - L >= 0) ? r - L : r - L + L2; // up
    return;
}
