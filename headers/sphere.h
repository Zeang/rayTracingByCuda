#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include <math.h>

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* mat) : center(cen), radius(r), mat_ptr(mat) {}
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 center;
	float radius;
	material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float t = (-b - sqrt(discriminant)) / a;
		if (t < tmax && t > tmin) {
			rec.t = t;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		t = (-b + sqrt(discriminant)) / a;
		if (t < tmax && t > tmin) {
			rec.t = t;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

#endif
