#include "renderer.h"
#include "scene.h"

#ifdef CUDA_ENABLED

#else
CUDA_HOSTDEV void Renderer::render(int i, int j, uint32_t* windowPixels, Camera* cam, hitable* world, Image* image, int sampleCount, uint8_t* fileOutputImage)
{
    RandomGenerator rng(sampleCount, i * image->nx + j);
    float u = float(i + rng.get1f()) / float(image->nx); // left to right
    float v = float(j + rng.get1f()) / float(image->ny); // bottom to top

    ray r = cam->getRay(rng, u, v);

    image->pixels[i][j] += color(rng, r, world, 0);

    vec3 col = image->pixels[i][j] / sampleCount;

    // Gamma encoding of images is used to optimize the usage of bits
    // when encoding an image, or bandwidth used to transport an image,
    // by taking advantage of the non-linear manner in which humans perceive
    // light and color. (wikipedia)

    // we use gamma 2: raising the color to the power 1/gamma (1/2)
    col = vec3(sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]));

    int ir = int(255.99f * col[0]);
    int ig = int(255.99f * col[1]);
    int ib = int(255.99f * col[2]);

    if (writeImagePNG || writeImagePPM)
    {
        // PNG
        int index = (image->ny - 1 - j) * image->nx + i;
        int index3 = 3 * index;

        fileOutputImage[index3 + 0] = ir;
        fileOutputImage[index3 + 1] = ig;
        fileOutputImage[index3 + 2] = ib;
    }

    if (showWindow)
        windowPixels[(image->ny - j - 1) * image->nx + i] = (ir << 16) | (ig << 8) | (ib);
}
#endif

CUDA_HOSTDEV bool Renderer::trace_rays(uint32_t* windowPixels, camera* cam, hitable* world, Image* image, int sampleCount, uint8_t* fileOutputImage)
{
#ifdef CUDA_ENABLED
    cudaRender(windowPixels, cam, world, image, sampleCount, fileOutputImage);
#else
    // collapses the two nested fors into the same parallel for
#pragma omp parallel for collapse(2)
    // j track rows - from top to bottom
    for (int j = 0; j < image->ny; ++j)
    {
        // i tracks columns - left to right
        for (int i = 0; i < image->nx; ++i)
        {
            render(i, j, windowPixels, cam, world, image, sampleCount, fileOutputImage);
        }
    }
#endif  // CUDA_ENABLED
    return true;
}