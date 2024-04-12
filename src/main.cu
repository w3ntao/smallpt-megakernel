#include <iostream>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>

#include "macro.h"
#include "sphere.h"
#include "mega_kernel.h"

#include "lodepng/lodepng.h"

void mage_kernel_rendering(Vec3 *frame_buffer, int width, int height, int num_samples, const Sphere* spheres, int num_spheres) {
    auto start = std::chrono::system_clock::now();

    const int thread_width = 8;
    const int thread_height = 8;

    dim3 threads(thread_width, thread_height);
    dim3 blocks(width / thread_width + 1, height / thread_height + 1);

    MegaKernel::render<<<blocks, threads>>>(frame_buffer, width, height, num_samples, spheres, num_spheres);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};
    std::cout << "rendering (" << num_samples << " spp) took " << std::fixed << std::setprecision(3)
              << duration.count() << " seconds. (block dimension: "
              << threads.x << "x" << threads.y << "x" << threads.z << ")\n" << std::flush;
}

int main() {
    const double ratio = 1.0;
    const int width = 1024 * ratio;
    const int height = 768 * ratio;

    const int num_samples = 64;

    Vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **) &frame_buffer, sizeof(Vec3) * width * height));

    const int num_spheres = 9;
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
                    ReflectionType::refractive); // Glass
    spheres[8].init(600, Vec3(50, 681.6 - .27, 81.6), Vec3(12, 12, 12), Vec3(0, 0, 0),
                    ReflectionType::diffuse); // Lite

    mage_kernel_rendering(frame_buffer, width, height,num_samples, spheres, num_spheres);

    std::vector<unsigned char> png_pixels(width * height * 4);

    for (int i = 0; i < width * height; i++) {
        png_pixels[4 * i + 0] = toInt(frame_buffer[i].x);
        png_pixels[4 * i + 1] = toInt(frame_buffer[i].y);
        png_pixels[4 * i + 2] = toInt(frame_buffer[i].z);
        png_pixels[4 * i + 3] = 255;
    }

    std::string file_name = "smallpt_megakernel_" + std::to_string(num_samples) + ".png";

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
