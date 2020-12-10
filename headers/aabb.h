#ifndef AABBH
#define AABBH

#include "ray.h"

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a < b ? b : a; }

class aabb {
public:
	__device__ aabb(){}
	__device__ aabb(const vec3& a, const vec3& b) { _min = a; _max = b; }
	__device__ vec3 min() const { return _min; }
	__device__ vec3 max() const { return _max; }

	__device__ bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; ++a) {
			float t0 = ffmin((_min[a] - r.origin()[a]) / r.direction()[a],
				(_max[a] - r.origin()[a]) / r.direction()[a]);
			float t1 = ffmax((_min[a] - r.origin()[a]) / r.direction()[a],
				(_max[a] - r.origin()[a]) / r.direction()[a]);
			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);
			if (tmin >= tmax)
				return false;
		}
		return true;
	}
	
	vec3 _min, _max;
};

#endif