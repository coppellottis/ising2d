#include "simulation.h"
#include <math.h>

// pre-computation of probabilites for Metropolis test
// J is set to 1.
void pre_prob(double* p, double beta) {
    p[1] = exp(-4*beta);
    p[2] = exp(-8*beta);
    return;
}