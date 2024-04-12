#pragma once

#include "sampler.h"
#include "util.h"

SMALLPT_GPU
inline int intersect(const Ray &r, double &t, const Sphere *spheres, int num_spheres) {
    int id = -1;
    t = std::numeric_limits<double>::infinity();

    for (int i = 0; i < num_spheres; ++i) {
        double d = spheres[i].intersect(r);
        if (d > 0 && d < t) {
            t = d;
            id = i;
        }
    }

    return id;
}

SMALLPT_GPU
Vec3 trace(const Ray &camera_ray, const Sphere *spheres, const int num_spheres, Sampler &sampler) {
    Vec3 radiance(0.0, 0.0, 0.0);
    Vec3 throughput(1.0, 1.0, 1.0);

    auto ray = camera_ray;

    for (int depth = 0;; ++depth) {
        double t; // distance to intersection
        int hit_sphere_id = intersect(ray, t, spheres, num_spheres);
        if (hit_sphere_id < 0) {
            break;
        }

        const Sphere &obj = spheres[hit_sphere_id]; // the hit object
        Vec3 hit_point = ray.o + ray.d * t;
        Vec3 surface_normal = (hit_point - obj.position).norm(); // always face out
        Vec3 normal = surface_normal.dot(ray.d) < 0 ? surface_normal : surface_normal * -1;

        radiance += throughput * obj.emission;
        throughput *= obj.color;

        if (depth > 4) {
            // russian roulette
            double probability_russian_roulette = clamp(throughput.max_component_val(), 0.1, 0.95);

            if (sampler.generate() >= probability_russian_roulette) {
                // terminated
                break;
            }
            // survive and enhanced
            throughput *= (1.0 / probability_russian_roulette);
        }

        if (obj.reflection_type == ReflectionType::diffuse) { // Ideal DIFFUSE reflection
            double r1 = 2 * M_PI * sampler.generate();
            double r2 = sampler.generate();
            double r2s = sqrt(r2);
            Vec3 w = normal;
            Vec3 u = (fabs(w.x) > 0.1 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)).cross(w).norm();
            Vec3 v = w.cross(u);
            Vec3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

            ray = Ray(hit_point, d);
            continue;
        }

        if (obj.reflection_type == ReflectionType::specular) { // Ideal SPECULAR reflection
            ray = Ray(hit_point, ray.d - surface_normal * 2 * surface_normal.dot(ray.d));
            continue;
        }

        Ray spawn_ray_reflect(hit_point,
                              ray.d - surface_normal * 2 *
                                      surface_normal.dot(ray.d)); // Ideal dielectric REFRACTION

        bool into = surface_normal.dot(normal) > 0; // Ray from outside going in?
        double nc = 1;
        double nt = 1.5;
        double nnt = into ? nc / nt : nt / nc;
        double ddn = ray.d.dot(normal);
        double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

        if (cos2t < 0) { // Total internal reflection
            ray = spawn_ray_reflect;
            continue;
        }

        Vec3 t_dir =
                (ray.d * nnt - surface_normal * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
        double a = nt - nc;
        double b = nt + nc;
        double R0 = a * a / (b * b);
        double c = 1 - (into ? -ddn : t_dir.dot(surface_normal));

        double Re = R0 + (1 - R0) * c * c * c * c * c;
        double Tr = 1 - Re;
        double probability_reflect = 0.25 + 0.5 * Re;

        double RP = Re / probability_reflect;
        double TP = Tr / (1 - probability_reflect);

        // refract or reflect
        if (sampler.generate() < probability_reflect) {
            // reflect
            ray = spawn_ray_reflect;
            throughput *= RP;
            continue;
        }

        // refract
        ray = Ray(hit_point, t_dir); // Ideal dielectric REFRACTION
        throughput *= TP;
        continue;
    }

    return radiance;
}

__global__
void render(Vec3 *frame_buffer, const int width, const int height, const int num_samples, const Sphere *spheres,
            const int num_spheres) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int flat_idx = (height - 1 - y) * width + x;

    Ray cam(Vec3(50, 52, 295.6), Vec3(0, -0.042612, -1).norm()); // cam pos, dir
    Vec3 cx = Vec3(width * 0.5135 / height, 0, 0);
    Vec3 cy = cx.cross(cam.d).norm() * 0.5135;

    Sampler sampler(flat_idx);

    auto pixel_val = Vec3(0.0, 0.0, 0.0);
    for (int s = 0; s < num_samples; s++) {
        double r1 = 2 * sampler.generate();
        double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);

        double r2 = 2 * sampler.generate();
        double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

        Vec3 d = cx * ((dx + x) / width - 0.5) + cy * ((dy + y - 20) / height - 0.5) + cam.d;

        pixel_val += trace(Ray(cam.o + d * 140, d.norm()), spheres, num_spheres, sampler);
    }

    pixel_val = pixel_val * (1.0 / double(num_samples));

    frame_buffer[flat_idx] = Vec3(clamp(pixel_val.x, 0, 1), clamp(pixel_val.y, 0, 1), clamp(pixel_val.z, 0, 1));
}
