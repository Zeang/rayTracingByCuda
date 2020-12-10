#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "vec3.h"
#include <math.h>

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* mat) : center(cen), radius(r), mat_ptr(mat) {}
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
	vec3 center;
	float radius;
	material* mat_ptr;
};

class moving_sphere : public hitable {
public:
	__device__ moving_sphere() {}
	__device__ moving_sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material* m)
		: center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
	__device__ vec3 center(float time) const;

	vec3 center0, center1;
	float time0, time1;
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

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {
	box = aabb(center - vec3(radius, radius, radius),
		center + vec3(radius, radius, radius));
	return true;
}

__device__ vec3 moving_sphere::center(float time) const {
	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	vec3 oc = r.origin() - center(r.time());
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < tmax && temp > tmin) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < tmax && temp > tmin) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

__device__ aabb surrounding_box(aabb box0, aabb box1) {
	vec3 small(ffmin(box0.min().x(), box1.min().x()),
		ffmin(box0.min().y(), box1.min().y()),
		ffmin(box0.min().z(), box1.min().z()));
	vec3 big(ffmax(box0.max().x(), box1.max().x()),
		ffmax(box0.max().y(), box1.min().y()),
		ffmax(box0.max().z(), box1.max().z()));
	return aabb(small, big);
}

__device__ bool moving_sphere::bounding_box(float t0, float t1, aabb& box) const {
	aabb box0(center(t0) - vec3(radius, radius, radius),
		center(t0) + vec3(radius, radius, radius));
	aabb box1(center(t1) - vec3(radius, radius, radius),
		center(t1) + vec3(radius, radius, radius));
	box = surrounding_box(box0, box1);
	return true;
}

#endif
