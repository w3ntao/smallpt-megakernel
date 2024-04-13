#pragma once

#include "ray.h"

enum class ReflectionType { diffuse, specular, refractive }; // material types, used in radiance()

struct Sphere {
    double radius; // radius

    Vec3 position;
    Vec3 emission;
    Vec3 color;

    ReflectionType reflection_type;

    void init(const double _radius, const Vec3 _position, const Vec3 _emission, const Vec3 _color,
              const ReflectionType _reflection_type) {
        radius = _radius;
        position = _position;
        emission = _emission;
        color = _color;
        reflection_type = _reflection_type;
    }

    SMALLPT_GPU double intersect(const Ray &r) const { // returns distance, 0 if nohit
        Vec3 op = position - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double epsilon = 1e-4;
        double b = op.dot(r.d);
        double determinant = b * b - op.dot(op) + radius * radius;
        if (determinant < 0) {
            return 0;
        }

        determinant = sqrt(determinant);

        if (double t = b - determinant; t > epsilon) {
            return t;
        }

        if (double t = b + determinant; t > epsilon) {
            return t;
        }

        return 0;
    }
};