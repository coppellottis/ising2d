#include <stdlib.h>
#include "wolff.h"
#include "lattice.h"
#include "metropolis.h"
#include "observables.h"
#include "pcg_basic.h"

static int create_cluster(Lattice* lattice, int* cluster, double p_add) {
    int L = lattice->L;
    char* in_cluster = calloc(L*L, sizeof(char));

    // One site r is chosen randomly
    int p = pcg32_boundedrand(L*L);
    int n_old = 0;
    int l = 1; // cluster length
    int n_new = l;

    cluster[0] = p;
    in_cluster[p] = 1;
    int spin = lattice->spins[p];

    while(n_new > n_old) {
        for(int i=n_old; i<n_new;i++) {
            int r = cluster[i];
            int nn[4]; 
            get_nn(lattice, r, nn); // get nearest neighbours, in observables.c
            for(int j = 0; j<4; j++) {
                if(!in_cluster[nn[j]] && lattice->spins[nn[j]]==spin) {
                    double t = (double)pcg32_random() / 4294967296.0;
                    if(t < p_add) {
                        cluster[l] = nn[j];
                        in_cluster[nn[j]] = 1;
                        l++;
                    }
            
                }
        
            }
        }
        n_old = n_new;
        n_new = l;
    }
    free(in_cluster);
    return l;
}

int wolff_update(Lattice* lattice, double p_add) {
    int L = lattice->L;
    int* cluster = calloc(L*L,sizeof(int));
    int l = create_cluster(lattice, cluster, p_add);

    for(int i = 0; i<l; i++) {
        lattice->spins[cluster[i]] *= -1;
    }

    free(cluster);
    return l;
}

// sweep_frac keeps track of the fraction of the lattice updated.
// When it reaches 1, the observables are measured
double wolff_sweep(Lattice* lattice, double p_add, double sweep_frac) {
    int L = lattice->L;
    while(sweep_frac < 1.0) {
        sweep_frac += (double)wolff_update(lattice, p_add)/(L*L);
    }
    return sweep_frac-1;
}