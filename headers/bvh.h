#ifndef BVHH
#define BVHH

#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "hitable.h"
#include "thrust/functional.h"
#include "thrust/sort.h"

class bvh_node : public hitable {
public:
	__device__ bvh_node() {}
	__device__ bvh_node(hitable** l, int n, float time0, float time1, curandState* local_rand_state);
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
	hitable* left;
	hitable* right;
	aabb box;
};

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
	b = box;
	return true;
}

__device__ bool bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	if (box.hit(r, tmin, tmax)) {
		hit_record left_rec, right_rec;
		bool hit_left = left->hit(r, tmin, tmax, left_rec);
		bool hit_right = right->hit(r, tmin, tmax, right_rec);
		if (hit_left && hit_right) {
			if (left_rec.t < right_rec.t)
				rec = left_rec;
			else
				rec = right_rec;
			return true;
		}
		else if (hit_left) {
			rec = left_rec;
			return true;
		}
		else if (hit_right) {
			rec = right_rec;
			return true;
		}
		else
			return false;
	}
	else
		return false;
}

__thrust_exec_check_disable__ __device__ bool box_x_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	hitable* ah = (hitable*)a;
	hitable* bh = (hitable*)b;

	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		printf("no bounding box in bvh_node constructor\n");

	if (box_left.min().y() - box_right.min().x() < 0.0f)
		return false;
	else
		return true;
}

__thrust_exec_check_disable__ __device__ bool box_y_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	hitable* ah = (hitable*)a;
	hitable* bh = (hitable*)b;

	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		printf("no bounding box in bvh_node constructor\n");

	if (box_left.min().y() - box_right.min().y() < 0.0f)
		return false;
	else
		return true;
}

__thrust_exec_check_disable__ __device__ bool box_z_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	hitable* ah = (hitable*)a;
	hitable* bh = (hitable*)b;

	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		printf("no bounding box in bvh_node constructor\n");
	if (box_left.min().z() - box_right.min().z() < 0.0f)
		return false;
	else
		return true;
}

__device__ bvh_node::bvh_node(hitable** l, int n, float time0, float time1, curandState* local_rand_state) {
	int axis = int(3 * curand_uniform(local_rand_state));
	switch (axis) {
	case 0:
		thrust::sort(l, l + n, box_x_compare);
		break;
	case 1:
		thrust::sort(l, l + n, box_y_compare);
		break;
	default:
		thrust::sort(l, l + n, box_z_compare);
	}
	switch (n) {
	case 0:
		left = right = l[0];
		break;
	case 1:
		left = l[0];
		right = l[1];
		break;
	default:
		left = new bvh_node(l, n / 2, time0, time1, local_rand_state);
		right = new bvh_node(l + n / 2, n - n / 2, time0, time1, local_rand_state);
	}
	
	aabb box_left, box_right;
	if (!left->bounding_box(time0, time1, box_left) ||
		!right->bounding_box(time0, time1, box_right)) {
		printf("no bounding box in bvh_node constructor\n");
	}
	box = surrounding_box(box_left, box_right);
}

#endif