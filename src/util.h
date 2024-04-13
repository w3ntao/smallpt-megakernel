#pragma once

#include "macro.h"

SMALLPT_CPU_GPU
static double clamp(double x, double low, double high) {
    return x < low ? low : x > high ? high : x;
}

static inline int toInt(double x) {
    return int(pow(clamp(x, 0, 1), 1 / 2.2) * 255 + .5);
}
