#include <stdio.h>
#include <stdlib.h>
#include "lattice.h"
#include "observables.h"



int main(void){

    Lattice* lat;
    lat = init_lattice(10,true);
    double m = get_magnetization(lat);
    free_lattice(lat);
    printf("Reticolo inizializzato!\n");
    printf("La magnetizzazione media iniziale è: %.2f\n", m);
    return 0;
}