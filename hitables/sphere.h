#pragma once

#include "hitable.h"
#include "../util/vec3.h"

class sphere : public hitable {
public:
	CUDA_DEV sphere() {}
	CUDA_DEV sphere(vec3 cen, float r, material* mat) : center(cen), radius(r), mat_ptr(mat) {}
	CUDA_DEV virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const
	{
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
	//CUDA_DEV virtual bool bounding_box(float t0, float t1, aabb& box) const;

	vec3 center;
	float radius;
	material* mat_ptr;
};

class moving_sphere : public hitable {
public:
	CUDA_DEV moving_sphere() {}
	CUDA_DEV moving_sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material* m)
		: center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {};
	CUDA_DEV virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const
	{
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
	//CUDA_DEV virtual bool bounding_box(float t0, float t1, aabb& box) const;
	CUDA_DEV vec3 center(float time) const
	{
		return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
	}

	vec3 center0, center1;
	float time0, time1;
	float radius;
	material* mat_ptr;
};

//CUDA_DEV bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
//	vec3 oc = r.origin() - center;
//	float a = dot(r.direction(), r.direction());
//	float b = dot(oc, r.direction());
//	float c = dot(oc, oc) - radius * radius;
//	float discriminant = b * b - a * c;
//	if (discriminant > 0) {
//		float t = (-b - sqrt(discriminant)) / a;
//		if (t < tmax && t > tmin) {
//			rec.t = t;
//			rec.p = r.point_at_parameter(rec.t);
//			rec.normal = (rec.p - center) / radius;
//			rec.mat_ptr = mat_ptr;
//			return true;
//		}
//		t = (-b + sqrt(discriminant)) / a;
//		if (t < tmax && t > tmin) {
//			rec.t = t;
//			rec.p = r.point_at_parameter(rec.t);
//			rec.normal = (rec.p - center) / radius;
//			rec.mat_ptr = mat_ptr;
//			return true;
//		}
//	}
//	return false;
//}

//CUDA_DEV bool sphere::bounding_box(float t0, float t1, aabb& box) const {
//	box = aabb(center - vec3(radius, radius, radius),
//		center + vec3(radius, radius, radius));
//	return true;
//}

//CUDA_DEV vec3 moving_sphere::center(float time) const {
//	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
//}

//CUDA_DEV bool moving_sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
//	vec3 oc = r.origin() - center(r.time());
//	float a = dot(r.direction(), r.direction());
//	float b = dot(oc, r.direction());
//	float c = dot(oc, oc) - radius * radius;
//	float discriminant = b * b - a * c;
//	if (discriminant > 0) {
//		float temp = (-b - sqrt(discriminant)) / a;
//		if (temp < tmax && temp > tmin) {
//			rec.t = temp;
//			rec.p = r.point_at_parameter(rec.t);
//			rec.normal = (rec.p - center(r.time())) / radius;
//			rec.mat_ptr = mat_ptr;
//			return true;
//		}
//		temp = (-b + sqrt(discriminant)) / a;
//		if (temp < tmax && temp > tmin) {
//			rec.t = temp;
//			rec.p = r.point_at_parameter(rec.t);
//			rec.normal = (rec.p - center(r.time())) / radius;
//			rec.mat_ptr = mat_ptr;
//			return true;
//		}
//	}
//	return false;
//}

//CUDA_DEV aabb surrounding_box(aabb box0, aabb box1) {
//	vec3 small(ffmin(box0.min().x(), box1.min().x()),
//		ffmin(box0.min().y(), box1.min().y()),
//		ffmin(box0.min().z(), box1.min().z()));
//	vec3 big(ffmax(box0.max().x(), box1.max().x()),
//		ffmax(box0.max().y(), box1.min().y()),
//		ffmax(box0.max().z(), box1.max().z()));
//	return aabb(small, big);
//}

//CUDA_DEV bool moving_sphere::bounding_box(float t0, float t1, aabb& box) const {
//	aabb box0(center(t0) - vec3(radius, radius, radius),
//		center(t0) + vec3(radius, radius, radius));
//	aabb box1(center(t1) - vec3(radius, radius, radius),
//		center(t1) + vec3(radius, radius, radius));
//	box = surrounding_box(box0, box1);
//	return true;
//}
