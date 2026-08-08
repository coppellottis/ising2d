
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
    int L = lattice->L;
    int right = (r % L == L - 1) ? r - L + 1 : r + 1;
    int left  = (r % L == 0) ? r + L - 1 : r - 1;
    int down  = (r + L) % (L*L);
    int up = (r - L + L*L) % (L*L);

    int S = lattice->spins[right]+lattice->spins[left]+lattice->spins[down]+lattice->spins[up];
    return S;
}