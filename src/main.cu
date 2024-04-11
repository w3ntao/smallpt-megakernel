#include <iostream>

#include "macro.h"
#include "lodepng/lodepng.h"

inline double clamp(double x, double low, double high) {
    return x < low ? low : x > high ? high : x;
}

inline int toInt(double x) { return int(pow(clamp(x, 0, 1), 1 / 2.2) * 255 + .5); }


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
    Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }

    SMALLPT_CPU_GPU
    void operator+=(const Vec3 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
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

enum class ReflectionType {
    diffuse, specular, Refractive
}; // material types, used in radiance()

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

__global__ void
render(Vec3 *frame_buffer, const int width, const int height, const Sphere *spheres, const int num_spheres) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= width * height) {
        return;
    }
    int y = worker_idx / width;
    int x = worker_idx % width;

    frame_buffer[worker_idx] = Vec3(double(x) / width, double(y) / height, 0.2);
}

int main() {
    const int width = 1024;
    const int height = 768;

    Vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **) &frame_buffer, sizeof(Vec3) * width * height));

    int num_spheres = 9;
    Sphere *spheres;
    checkCudaErrors(cudaMallocManaged((void **) &spheres, sizeof(Sphere) * num_spheres));

    spheres[0].init(1e5, Vec3(1e5 + 1, 40.8, 81.6), Vec3(0, 0, 0), Vec3(.75, .25, .25),
                    ReflectionType::diffuse); // Left
    spheres[1].init(1e5, Vec3(-1e5 + 99, 40.8, 81.6), Vec3(0, 0, 0), Vec3(.25, .25, .75),
                    ReflectionType::diffuse);  // Right

    spheres[2].init(1e5, Vec3(50, 40.8, 1e5), Vec3(0, 0, 0), Vec3(.75, .75, .75),
                    ReflectionType::diffuse); // Back

    spheres[3].init(1e5, Vec3(50, 40.8, -1e5 + 170), Vec3(0, 0, 0), Vec3(0, 0, 0),
                    ReflectionType::diffuse); // Front
    spheres[4].init(1e5, Vec3(50, 1e5, 81.6), Vec3(0, 0, 0), Vec3(.75, .75, .75),
                    ReflectionType::diffuse); // Bottom
    spheres[5].init(1e5, Vec3(50, -1e5 + 81.6, 81.6), Vec3(0, 0, 0), Vec3(.75, .75, .75),
                    ReflectionType::diffuse); // Top
    spheres[6].init(16.5, Vec3(27, 16.5, 47), Vec3(0, 0, 0), Vec3(1, 1, 1) * .999,
                    ReflectionType::specular); // Mirror
    spheres[7].init(16.5, Vec3(73, 16.5, 78), Vec3(0, 0, 0), Vec3(1, 1, 1) * .999,
                    ReflectionType::Refractive); // Glass
    spheres[8].init(600, Vec3(50, 681.6 - .27, 81.6), Vec3(12, 12, 12), Vec3(0, 0, 0),
                    ReflectionType::diffuse); // Lite

    const int thread_size = 64;
    dim3 threads(thread_size);
    dim3 blocks(width * height / thread_size + 1);
    render<<<blocks, threads>>>(frame_buffer, width, height, spheres, num_spheres);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::vector<unsigned char> png_pixels(width * height * 4);

    for (int i = 0; i < width * height; i++) {
        png_pixels[4 * i + 0] = toInt(frame_buffer[i].x);
        png_pixels[4 * i + 1] = toInt(frame_buffer[i].y);
        png_pixels[4 * i + 2] = toInt(frame_buffer[i].z);
        png_pixels[4 * i + 3] = 255;
    }

    std::string file_name = "small_pt_megakernel.png";
    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(file_name, png_pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());

    checkCudaErrors(cudaFree(frame_buffer));
    checkCudaErrors(cudaFree(spheres));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}
