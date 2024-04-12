#pragma once

#include "macro.h"

struct Vec3 {
    double x, y, z;

    SMALLPT_CPU_GPU
    Vec3() : x(0), y(0), z(0) {}

    SMALLPT_CPU_GPU
    Vec3(double _x, double _y, double _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    SMALLPT_CPU_GPU
    Vec3 abs() const {
        return Vec3(::abs(x), ::abs(y), ::abs(z));
    }

    SMALLPT_CPU_GPU
    bool is_positive() const {
        return x > 0 && y > 0 && z > 0;
    }

    SMALLPT_CPU_GPU
    Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }

    SMALLPT_CPU_GPU
    void operator+=(const Vec3 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
    }

    SMALLPT_CPU_GPU
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    SMALLPT_CPU_GPU
    Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }

    SMALLPT_CPU_GPU
    Vec3 operator*(double b) const { return Vec3(x * b, y * b, z * b); }

    SMALLPT_CPU_GPU
    void operator*=(double b) {
        x *= b;
        y *= b;
        z *= b;
    }

    SMALLPT_CPU_GPU
    Vec3 operator*(const Vec3 &b) const { return Vec3(x * b.x, y * b.y, z * b.z); }

    SMALLPT_CPU_GPU
    void operator*=(const Vec3 &b) {
        x *= b.x;
        y *= b.y;
        z *= b.z;
    }

    SMALLPT_CPU_GPU
    Vec3 norm() const { return *this * (1 / sqrt(x * x + y * y + z * z)); }

    SMALLPT_CPU_GPU
    double dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }

    SMALLPT_CPU_GPU
    Vec3 cross(const Vec3 &b) const {
        return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }

    SMALLPT_CPU_GPU
    double max_component_val() const { return x > y && x > z ? x : y > z ? y : z; }
};

struct Ray {
    Vec3 o, d;

    SMALLPT_CPU_GPU
    Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_) {}
};
