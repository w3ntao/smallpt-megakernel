#pragma once

#include <vector>
#include "sphere.h"
#include "sampler.h"
#include "util.h"

namespace WaveFront {
    struct Intersection {
        Vec3 position;
        Vec3 normal;
        bool intersected;
    };

    SMALLPT_GPU
    void intersect(Intersection &intersection, const Ray &r, const Sphere *spheres, int num_spheres) {
        double t = std::numeric_limits<double>::infinity();

        int id = -1;
        for (int i = 0; i < num_spheres; ++i) {
            double _t_hit = spheres[i].intersect(r);
            if (_t_hit > 0 && _t_hit < t) {
                t = _t_hit;
                id = i;
            }
        }

        if (id < 0) {
            intersection.intersected = false;
            return;
        }

        intersection.position = r.o + r.d * t;
        intersection.normal = (intersection.position - spheres[id].position).norm();
    }

    struct Queues {
        uint32_t *new_paths;
        uint32_t length_new_path;

        uint32_t *primary_ray_cast;
        uint32_t length_primiry_rays;
    };

    struct PathState {
        Ray *primary_rays;
        Intersection *intersections;
        Vec3 *radiance;
        Vec3 *throughput;
        uint32_t *path_length;
        Sampler *samplers;

        Vec3 *film_radiance;

        uint width;
        uint height;
        uint num;
    };

    __global__
    void init_path_state(PathState *path_state) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= path_state->num) {
            return;
        }

        path_state->path_length[worker_idx] = 0;
        path_state->intersections[worker_idx].intersected = false;

        path_state->throughput[worker_idx] = Vec3(0, 0, 0);
        path_state->radiance[worker_idx] = Vec3(0, 0, 0);

        path_state->film_radiance[worker_idx] = Vec3(0, 0, 0);

        path_state->samplers[worker_idx].init(worker_idx);
    }

    __global__
    void logic_kernel(PathState *path_state, Queues *queues) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= path_state->num) {
            return;
        }

        if (!path_state->throughput[worker_idx].is_positive()) {
            // when throughput drops below 0: time to generate a new path

            if (path_state->radiance[worker_idx].is_positive()) {
                // accumulate radiance
                path_state->film_radiance[worker_idx] += path_state->radiance[worker_idx];
            }


            uint32_t queue_id = atomicAdd(&queues->length_new_path, 1);
            queues->new_paths[queue_id] = worker_idx;
            // add to new_path queue
            return;
        }

        // TODO: implement me
    }

    __global__
    void new_path_kernel(PathState *path_state, Queues *queues) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= queues->length_new_path) {
            return;
        }

        const uint width = path_state->width;
        const uint height = path_state->height;

        const uint path_idx = queues->new_paths[worker_idx];

        const uint y = height - 1 - path_idx / width;
        const uint x = path_idx % width;

        Ray cam(Vec3(50, 52, 295.6), Vec3(0, -0.042612, -1).norm()); // cam pos, dir
        Vec3 cx = Vec3(width * 0.5135 / height, 0, 0);
        Vec3 cy = cx.cross(cam.d).norm() * 0.5135;

        Sampler &sampler = path_state->samplers[worker_idx];

        double r1 = 2 * sampler.generate();
        double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);

        double r2 = 2 * sampler.generate();
        double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

        Vec3 d = cx * ((dx + x) / width - 0.5) + cy * ((dy + y - 20) / height - 0.5) + cam.d;

        path_state->primary_rays[path_idx].o = cam.o;
        path_state->primary_rays[path_idx].d = d;

        path_state->radiance[path_idx] = Vec3(0, 0, 0);
        path_state->throughput[path_idx] = Vec3(1, 1, 1);

        auto queue_idx = atomicAdd(&queues->length_primiry_rays, 1);
        queues->primary_ray_cast[queue_idx] = path_idx;
    }

    __global__
    void primary_ray_cast_kernel(PathState *path_state, Queues *queues) {
        //TODO: progress 2024/04/12 implementing primary_ray_cast_kernel()
    }

    void render(Vec3 *frame_buffer, int width, int height, int num_samples, const Sphere *spheres,
                int num_spheres) {

        const uint path_num = width * height;

        PathState *path_state;
        checkCudaErrors(cudaMallocManaged((void **) &path_state, sizeof(PathState)));

        checkCudaErrors(cudaMallocManaged((void **) &path_state->primary_rays, sizeof(Ray) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &path_state->intersections, sizeof(Intersection) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &path_state->radiance, sizeof(Vec3) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &path_state->film_radiance, sizeof(Vec3) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &path_state->throughput, sizeof(Vec3) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &path_state->path_length, sizeof(uint32_t) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &path_state->samplers, sizeof(Sampler) * path_num));

        path_state->num = path_num;
        path_state->width = width;
        path_state->height = height;

        Queues *queues;
        checkCudaErrors(cudaMallocManaged((void **) &queues, sizeof(Queues)));
        checkCudaErrors(cudaMallocManaged((void **) &queues->new_paths, sizeof(uint32_t) * path_num));
        checkCudaErrors(cudaMallocManaged((void **) &queues->primary_ray_cast, sizeof(uint32_t) * path_num));

        int thread_size = 256;
        init_path_state<<<path_num / thread_size + 1, thread_size >>>(path_state);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        for (int s = 0; s < num_samples; s++) {
            queues->length_new_path = 0;
            queues->length_primiry_rays = 0;

            logic_kernel<<<path_num / thread_size + 1, thread_size>>>(path_state, queues);
            checkCudaErrors(cudaDeviceSynchronize());
            //checkCudaErrors(cudaGetLastError());

            new_path_kernel<<<queues->length_new_path / thread_size + 1, thread_size>>>(path_state, queues);
            checkCudaErrors(cudaDeviceSynchronize());
            //checkCudaErrors(cudaGetLastError());

            primary_ray_cast_kernel<<<queues->length_primiry_rays / thread_size + 1, thread_size>>>(path_state, queues);
            checkCudaErrors(cudaDeviceSynchronize());

            break;
        }

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        for (auto ptr: std::vector<void *>(
                {path_state->primary_rays, path_state->intersections, path_state->radiance, path_state->film_radiance,
                 path_state->throughput, path_state->path_length})) {
            checkCudaErrors(cudaFree(ptr));
        }
        checkCudaErrors(cudaFree(path_state));

        for (auto ptr: std::vector<void *>({queues->new_paths})) {
            checkCudaErrors(cudaFree(ptr));
        }
        checkCudaErrors(cudaFree(queues));
    }

}