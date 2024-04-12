#pragma once

#include "macro.h"

struct Sampler {
    SMALLPT_GPU void init(uint seed) {
        curand_init(seed, 0, 0, &rand_state);
    }

    SMALLPT_GPU
    inline double generate() { return curand_uniform(&rand_state); }

private:
    curandState rand_state;
};