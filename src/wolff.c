#include <stdlib.h>
#include "wolff.h"
#include "lattice.h"
#include "metropolis.h"
#include "observables.h"
#include "pcg_basic.h"

static int is_in_cluster(int* haystack, int needle, int l) {
    for(int i = 0; i<l; i++) {
        if(haystack[i]==needle) return 1;
    }
    return 0;
}

static void create_cluster(Lattice* lattice, int* cluster, double P_add) {
    int L = lattice->L;
    char* in_cluster = calloc(L*L, sizeof(char));

    // One site r is chosen randomly
    int p = pcg32_boundedrand(L*L);
    int n_old = 0;
    int l = 1; // cluster length
    int n_new = l;

    cluster[0] = p;
    int spin = lattice->spins[p];

    while(n_new > n_old) {
        for(int i=n_old; i<n_new;i++) {
            int r = cluster[i];
            int nn[4];
            get_nn(lattice, r, nn);
            for(int j = 0; j<4; j++) {
                if(!is_in_cluster(cluster, nn[j], l) && lattice->spins[nn[j]]==spin) {
                    double t = (double)pcg32_random() / 4294967296.0;
                    if(t < P_add) {
                        cluster[l] = nn[j];
                        l++;
                
                    }
            
                }
        
            }
        }
        n_old = n_new;
        n_new = l;
    }
    return;
}

void wolff_update(Lattice* lattice) {
    return;
}

void wolff_sweep(Lattice* lattice) {
    return;
}