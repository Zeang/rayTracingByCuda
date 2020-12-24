#pragma once

#include <iostream>
#include <random>
#include <float.h>
#include <omp.h>

#include "camera.h"
#include "image.h"
#include "randomGenerator.h"
#include "../materials/material.h"
#include "../hitables/sphere.h"
#include "../hitables/hitable_list.h"

class Renderer
{
	bool showWindow;
	bool writeImagePPM;
	bool writeImagePNG;

public:
	CUDA_HOSTDEV Renderer(bool showWindow, bool writeImagePPM, bool writeImagePNG) : showWindow(showWindow), writeImagePPM(writeImagePPM), writeImagePNG(writeImagePNG) {}

	CUDA_DEV vec3 color(RandomGenerator& rng, const ray& r, hitable* world, int depth)
	{
		ray cur_ray = r;
		vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
        for (int i = 0; i < 50; ++i) {
            hit_record rec;
            if (world->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
                ray scattered;
                vec3 attenuation;
                if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, cur_ray, rng)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0, 0, 0);
                }
            }
            else {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float t = 0.5f * (unit_direction.y() + 1.0f);
                vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        return vec3(0.0f, 0.0f, 0.0f); // exceeded recursion
	}

    CUDA_HOSTDEV bool trace_rays(uint32_t* windowPixels, camera* cam, hitable* world, Image* image, int simpleCount, uint8_t* fileOutputImage);

#ifdef CUDA_ENABLED
    void cudaRender(uint32_t* windowPixels, camera* cam, hitable* world, Image* image, int sampleCount, uint8_t* fileOutputImage);
#else
    CUDA_HOSTDEV void render(int i, int j, uint32_t* windowPixels, Camera* cam, hitable* world, Image* image, int sampleCount, uint8_t* fileOutputImage);
#endif  // CUDA_ENABLED
};