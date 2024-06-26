#include <iostream>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>

#include "macro.h"
#include "sphere.h"
#include "megakernel.h"
#include "wavefront.h"

#include "lodepng/lodepng.h"

int main() {
    const double ratio = 1.0;
    const int width = 1024 * ratio;
    const int height = 768 * ratio;

    const int num_samples = 64;

    Vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, sizeof(Vec3) * width * height));

    const int num_spheres = 9;
    Sphere *spheres;
    checkCudaErrors(cudaMallocManaged((void **)&spheres, sizeof(Sphere) * num_spheres));

    spheres[0].init(1e5, Vec3(1e5 + 1, 40.8, 81.6), Vec3(0, 0, 0), Vec3(.75, .25, .25),
                    ReflectionType::diffuse); // Left
    spheres[1].init(1e5, Vec3(-1e5 + 99, 40.8, 81.6), Vec3(0, 0, 0), Vec3(.25, .25, .75),
                    ReflectionType::diffuse); // Right

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

    auto start = std::chrono::system_clock::now();

    std::string file_name;

    if (true) {
        WaveFront::render(frame_buffer, width, height, num_samples, spheres, num_spheres);
        file_name = "smallpt_wavefront_" + std::to_string(num_samples) + ".png";
    } else {
        MegaKernel::render(frame_buffer, width, height, num_samples, spheres, num_spheres);
        file_name = "smallpt_megakernel_" + std::to_string(num_samples) + ".png";
    }

    const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};
    std::cout << "rendering (" << num_samples << " spp) took " << std::fixed << std::setprecision(3)
              << duration.count() << " seconds\n"
              << std::flush;

    std::vector<unsigned char> png_pixels(width * height * 4);

    for (int i = 0; i < width * height; i++) {
        png_pixels[4 * i + 0] = toInt(frame_buffer[i].x);
        png_pixels[4 * i + 1] = toInt(frame_buffer[i].y);
        png_pixels[4 * i + 2] = toInt(frame_buffer[i].z);
        png_pixels[4 * i + 3] = 255;
    }

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
