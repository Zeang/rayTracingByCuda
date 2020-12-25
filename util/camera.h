#pragma once

#include "ray.h"
#include "util.h"
#include "randomGenerator.h"

enum cameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// fov - field of view
// image is not square => fow is different horizontally and vertically

class camera
{
public:
    vec3 origin;
    vec3 lowerLeftCorner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lensRadius;

    vec3 lookFrom;
    vec3 lookAt;

    vec3 vup;
    float vfov;
    float aspect;
    float aperture;
    float focusDist;

    float halfWidth;
    float halfHeight;

    CUDA_HOSTDEV camera() 
      : lowerLeftCorner(vec3(-2.0f, -1.0f, -1.0f)),
        horizontal(vec3(4.0f, 0.0f, 0.0f)),
        vertical(vec3(0.0f, 2.0f, 0.0f)),
        origin(vec3(0.0f, 0.0f, 0.0f)) {};

    CUDA_HOSTDEV camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect)
    {
        float theta = vfov * M_PI / 180.f;
        this->halfHeight = tan(theta / 2.0f);
        this->halfWidth = aspect * halfHeight;

        this->origin = lookFrom;
        this->w = unit_vector(lookFrom - lookAt);
        this->u = unit_vector(cross(vup, w));
        this->v = cross(w, u);

        this->lowerLeftCorner = origin - halfWidth * u - halfHeight * v - w;
        this->horizontal = 2.0f * halfWidth * u;
        this->vertical = 2.0f * halfHeight * v;

        this->lookFrom = lookFrom;
        this->lookAt = lookAt;

        this->vup = unit_vector(vup);
        this->vfov = vfov;
        this->aspect = aspect;
    }

    // another constructor
    CUDA_HOSTDEV camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect, float focusDist, float aperture = 0.0f) :
        camera(lookFrom, lookAt, vup, vfov, aspect)
    {
        this->lensRadius = aperture / 2.0f;
        this->aperture = aperture;
        this->focusDist = focusDist;
    }

    CUDA_HOSTDEV void update()
    {
        float theta = vfov * M_PI / 180.0f;
        this->halfHeight = tan(theta / 2.0f);
        this->halfWidth = aspect * halfHeight;

        this->origin = lookFrom;
        this->w = unit_vector(lookFrom - lookAt);
        this->u = unit_vector(cross(vup, w));
        this->v = cross(w, u);

        this->lowerLeftCorner = origin - halfWidth * focusDist * u - halfHeight * focusDist * v - focusDist * w;
        this->horizontal = 2.0f * halfWidth * focusDist * u;
        this->vertical = 2.0f * halfHeight * focusDist * v;
    }

    // Spherical coordinate system implementation - rotate the lookFrom location by theta polar angle and phi azimuth angle - keeping the distance 
    CUDA_HOSTDEV void rotate(float theta, float phi)
    {
        float radialDistance = (lookFrom - lookAt).length();
        this->lookFrom = vec3(
            radialDistance * sinf(theta) * sinf(phi),
            radialDistance * cosf(theta),
            radialDistance * sinf(theta) * cosf(phi)) + lookAt;
        update();
    }

    CUDA_HOSTDEV void zoom(float zoomScale)
    {
        this->vfov += zoomScale;
        // min(max())
        this->vfov = clamp<float>(this->vfov, 0.0f, 180.0f);
        update();
    }

    CUDA_HOSTDEV void translate(cameraMovement direction, float stepScale)
    {
        if (direction == FORWARD)
        {
            lookFrom += this->v * stepScale;
            lookAt += this->v * stepScale;
        }
        if (direction == BACKWARD)
        {
            lookFrom -= this->v * stepScale;
            lookAt -= this->v * stepScale;
        }
        if (direction == LEFT)
        {
            lookFrom -= this->u * stepScale;
            lookAt -= this->u * stepScale;
        }
        if (direction == RIGHT)
        {
            lookFrom += this->u * stepScale;
            lookAt += this->u * stepScale;
        }
        update();
    }

    CUDA_DEV ray get_ray(RandomGenerator& rng, float s, float t)
    {
        vec3 rd = lensRadius * rng.random_in_unit_sphere();
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset);
    }
};

//__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
//    vec3 p;
//    do {
//        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
//    } while (dot(p, p) >= 1.0f);
//    return p;
//}
//
//class camera {
//public:
//    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) {
//        time0 = t0;
//        time1 = t1;
//        lens_radius = aperture / 2.0f;
//        float theta = vfov * ((float)M_PI) / 180.0f;
//        float half_height = tan(theta / 2.0f);
//        float half_width = aspect * half_height;
//        origin = lookfrom;
//        w = unit_vector(lookfrom - lookat);
//        u = unit_vector(cross(vup, w));
//        v = cross(w, u);
//        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
//        horizontal = 2.0f * half_width * focus_dist * u;
//        vertical = 2.0f * half_height * focus_dist * v;
//    }
//
//    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
//        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
//        vec3 offset = u * rd.x() + v * rd.y();
//        float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
//        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, time);
//    }
//
//	vec3 origin;
//	vec3 lower_left_corner;
//	vec3 horizontal;
//	vec3 vertical;
//    vec3 u, v, w;
//    float time0, time1;
//    float lens_radius;
//};
