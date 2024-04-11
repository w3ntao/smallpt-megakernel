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

__global__ void render(Vec3 *frame_buffer, const int width, const int height) {
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

    const int thread_size = 64;
    dim3 threads(thread_size);
    dim3 blocks(width * height / thread_size + 1);
    render<<<blocks, threads>>>(frame_buffer, width, height);
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
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();
    
    return 0;
}
