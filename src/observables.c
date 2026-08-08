
#include "observables.h"
#include "lattice.h"

double get_magnetization(Lattice* lattice) {
    double M = 0;
    int L = lattice->L;
    for(int i=0; i < L*L; i++) {
        M = M+lattice->spins[i];
    }
    double m = M/(L*L);
    return m;
}