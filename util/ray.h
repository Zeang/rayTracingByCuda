#ifndef RAYH
#define RAYH

#include "vec3.h"

class ray {
public:
	CUDA_HOSTDEV ray() {}
	CUDA_HOSTDEV ray(const vec3& a, const vec3& b, float ti = 0.0f) { A = a; B = b; _time = ti; }
	CUDA_HOSTDEV vec3 origin() const { return A; }
	CUDA_HOSTDEV vec3 direction() const { return B; }
	CUDA_HOSTDEV float time() const { return _time; }
	CUDA_HOSTDEV vec3 point_at_parameter(float t) const { return A + t * B; }

	vec3 A;
	vec3 B;
	float _time;
};

#endif