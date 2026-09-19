#ifndef WOLFF_H
    #define WOLFF_H 1

    #include "lattice.h"
    #include "pcg_basic.h"

    // Scratch buffers reused across every Wolff cluster update for a given
    // lattice. Previously "cluster" and "in_cluster" were malloc/calloc'd
    // and freed on *every single* wolff_update() call -- for a size-L
    // lattice that's two O(L^2) heap allocations per cluster flip, and
    // wolff_sweep() calls wolff_update() repeatedly (many times away from
    // criticality, where clusters are small) until it has flipped ~L^2
    // spins. Allocating these once per (L, beta) run and reusing them
    // removes that allocator churn entirely.
    //
    // "stamp"/"stamp_val" replace the old calloc'd in_cluster[] char array:
    // a site is "in the current cluster" iff stamp[site] == stamp_val.
    // Starting a new cluster just increments stamp_val (O(1)) instead of
    // re-zeroing an O(L^2) array.
    typedef struct {
        int* cluster;   // size L*L: sites belonging to the current cluster
        int* stamp;     // size L*L: last stamp_val at which each site was added
        int stamp_val;  // current "in cluster" marker
    } WolffBuffers;

    WolffBuffers* wolff_buffers_init(int L);
    void wolff_buffers_free(WolffBuffers* buf);

    int wolff_update(Lattice* lattice, double p_add, WolffBuffers* buf, pcg32_random_t* rng);
    double wolff_sweep(Lattice* lattice, double p_add, double sweep_frac, WolffBuffers* buf, pcg32_random_t* rng);

#endif
