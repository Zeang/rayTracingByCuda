#pragma once

#include <float.h>

#include "../hitables/sphere.h"
#include "../hitables/hitable_list.h"
#include "../materials/material.h"
#include "../util/randomGenerator.h"
#include "../util/common.h"

#ifdef CUDA_ENABLED
CUDA_GLOBAL void simpleScene(hitable** list, hitable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
		list[1] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
		list[2] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
		list[3] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

		*world = new hitable_list(list, 4);
	}	
}
CUDA_GLOBAL void simpleScene2(hitable** list, hitable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
        RandomGenerator rng;

        int count = 58;
        list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
        list[1] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
        list[2] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
        list[3] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));
        int i = 4;

        for (int a = -3; a < 3; a++)
        {
            for (int b = -4; b < 5; b++)
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

        *world = new hitable_list(list, count);
	}
}

CUDA_GLOBAL void randomScene(hitable** list, hitable** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        RandomGenerator rng;

        int n = 901;
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

        *world = new hitable_list(list, n);
    }
}

CUDA_GLOBAL void randomScene2(hitable** list, hitable** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        RandomGenerator rng;

        int n = 102;
        list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
        int i = 1;
        for (int a = -5; a < 5; a++)
        {
            for (int b = -5; b < 5; b++)
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
        list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.3f, 0.0f, 0.0f)));
        list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.4f, 0.5f, 0.6f), 0.0f));

        *world = new hitable_list(list, n);
    }
}

CUDA_GLOBAL void randomScene3(hitable** list, hitable** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        RandomGenerator rng;

        int n = 68;
        list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
        int i = 1;
        for (int a = -4; a < 4; a++)
        {
            for (int b = -4; b < 4; b++)
            {
                float chooseMat = rng.get1f();
                vec3 center(a + 0.9f * rng.get1f(), 0.2f, b + 0.9f * rng.get1f());
                if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
                {
                    if (chooseMat < 0.3)            // diffuse
                    {
                        list[i++] = new sphere(center, 0.2f, new lambertian(vec3(rng.get1f() * rng.get1f(), rng.get1f() * rng.get1f(), rng.get1f() * rng.get1f())));
                    }
                    else if (chooseMat < 0.65)      // metal
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
        list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.3f, 0.0f, 0.0f)));
        list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.4f, 0.5f, 0.6f), 0.0f));

        *world = new hitable_list(list, n);
    }
}

#endif  // CUDA_ENABLED
