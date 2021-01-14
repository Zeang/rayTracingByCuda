#pragma once

#include <float.h>

#include "../hitables/sphere.h"
#include "../hitables/hitable_list.h"
#include "../materials/material.h"
#include "randomGenerator.h"
#include "common.h"

CUDA_HOSTDEV inline hitable* simpleScene()
{
	hitable** list = new hitable * [4];
	list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
	list[1] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
	list[2] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
	list[3] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

	return new hitable_list(list, 4);
}

CUDA_HOSTDEV inline hitable* simpleScene2()
{
	RandomGenerator rng;

    int count = 20;
    hitable** list = new hitable * [count];
    list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
    list[1] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    list[2] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
    list[3] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));
    int i = 4;

    for (int a = -2; a < 2; a++)
    {
        for (int b = -2; b < 2; b++)
        {
            float chooseMat = rng.get1f();
            vec3 center(a + 0.9f * rng.get1f(), 0.2f, b + 0.9f * rng.get1f());
            if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
            {
                if (chooseMat < 0.5)            // diffuse
                {
                    list[i++] = new sphere(center, 0.2f, new lambertian(vec3(rng.get1f() * rng.get1f(), rng.get1f() * rng.get1f(), rng.get1f() * rng.get1f())));
                }
                else if (chooseMat < 0.75)      // metal
                {
                    list[i++] = new sphere(center, 0.2f, new metal(vec3(0.5 * (1 + rng.get1f()), 0.5 * (1 + rng.get1f()), 0.5 * (1 + rng.get1f()))));
                }
                else                            // glass
                {
                    list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
                }
            }
        }
    }

    return new hitable_list(list, count);
}

inline hitable* randomScene()
{
    RandomGenerator rng;

    int n = 1000;
    hitable** list = new hitable * [n + 1];
    list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
    int i = 1;
    for (int a = -15; a < 15; a++)
    {
        for (int b = -15; b < 15; b++)
        {
            float chooseMat = rng.get1f();
            vec3 center(a + 0.9f * rng.get1f(), 0.2f, b + 0.9f * rng.get1f());
            if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
            {
                if (chooseMat < 0.5)            // diffuse
                {
                    list[i++] = new sphere(center, 0.2f, new lambertian(vec3(rng.get1f() * rng.get1f(), rng.get1f() * rng.get1f(), rng.get1f() * rng.get1f())));
                }
                else if (chooseMat < 0.75)      // metal
                {
                    list[i++] = new sphere(center, 0.2f, new metal(vec3(0.5 * (1 + rng.get1f()), 0.5 * (1 + rng.get1f()), 0.5 * (1 + rng.get1f()))));
                }
                else                            // glass
                {
                    list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
    list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    return new hitable_list(list, i);
}