#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "wolff.h"
#include "lattice.h"
#include "metropolis.h"
#include "observables.h"
#include "pcg_basic.h"

WolffBuffers* wolff_buffers_init(int L) {
    WolffBuffers* buf = malloc(sizeof(WolffBuffers));
    buf->cluster = malloc(L*L*sizeof(int));
    buf->stamp = calloc(L*L, sizeof(int)); // 0 = "not in any cluster yet"
    buf->stamp_val = 0;
    return buf;
}

void wolff_buffers_free(WolffBuffers* buf) {
    free(buf->cluster);
    free(buf->stamp);
    free(buf);
}

static int create_cluster(Lattice* lattice, double p_add, WolffBuffers* buf, pcg32_random_t* rng) {
    int L = lattice->L;
    int* cluster = buf->cluster;
    int* stamp = buf->stamp;

    // Guard against the (practically unreachable, but cheap to handle)
    // wraparound of a 32-bit counter after ~2^31 cluster builds.
    if (buf->stamp_val == INT_MAX) {
        memset(stamp, 0, L*L*sizeof(int));
        buf->stamp_val = 0;
    }
    int mark = ++buf->stamp_val;

    // One site r is chosen randomly
    int p = pcg32_boundedrand_r(rng, L*L);
    int n_old = 0;
    int l = 1; // cluster length
    int n_new = l;

    cluster[0] = p;
    stamp[p] = mark;
    int spin = lattice->spins[p];

    while(n_new > n_old) {
        for(int i=n_old; i<n_new;i++) {
            int r = cluster[i];
            int nn[4];
            get_nn(lattice, r, nn); // get nearest neighbours, in observables.c
            for(int j = 0; j<4; j++) {
                if(stamp[nn[j]] != mark && lattice->spins[nn[j]]==spin) {
                    double t = (double)pcg32_random_r(rng) / 4294967296.0;
                    if(t < p_add) {
                        cluster[l] = nn[j];
                        stamp[nn[j]] = mark;
                        l++;
                    }

                }

            }
        }
        n_old = n_new;
        n_new = l;
    }
    return l;
}

int wolff_update(Lattice* lattice, double p_add, WolffBuffers* buf, pcg32_random_t* rng) {
    int l = create_cluster(lattice, p_add, buf, rng);

    for(int i = 0; i<l; i++) {
        lattice->spins[buf->cluster[i]] *= -1;
    }

    return l;
}

// sweep_frac keeps track of the fraction of the lattice updated.
// When it reaches 1, the observables are measured
double wolff_sweep(Lattice* lattice, double p_add, double sweep_frac, WolffBuffers* buf, pcg32_random_t* rng) {
    int L = lattice->L;
    while(sweep_frac < 1.0) {
        sweep_frac += (double)wolff_update(lattice, p_add, buf, rng)/(L*L);
    }
    return sweep_frac-1;
}
