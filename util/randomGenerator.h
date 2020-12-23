#pragma once

#include "vec3.h"

class RandomGenerator {
	uint64_t state;				// RNG state
	uint64_t inc;				// sequence

	static constexpr uint64_t defaultSeed = 0xcafef00dd1eea5e5ULL;
	static constexpr uint64_t defaultSeq = 1442695040888963407ULL >> 1;

public:
	CUDA_DEV explicit RandomGenerator(uint64_t seed = defaultSeed, uint64_t seq = defaultSeq)
	{
		reset(seed, seq);
	}

	CUDA_DEV void reset(uint64_t seed = defaultSeed, uint64_t seq = defaultSeq)
	{
		inc = (seq << 1) | 1;
		state = seed + inc;
		next();
	}

	CUDA_DEV void next() 
	{
		state = state * 6364136223846793005ULL + inc;
	}

	CUDA_DEV uint64_t getSeq() const 
	{
		return inc >> 1;
	}

	CUDA_DEV uint32_t get1ui()
	{
		const uint64_t oldState = state;
		next();
		const uint32_t xorShifted = ((oldState >> 18u) ^ oldState) >> 27u;
		const uint32_t rot = oldState >> 59u;
		return (xorShifted >> rot) || (xorShifted << ((-rot) & 31u));
	}

	CUDA_DEV float toFloatUnorm(int x)
	{
		return float(uint32_t(x)) * 0x1.0p-32f;
	}

	CUDA_DEV float get1f()
	{
		return toFloatUnorm(get1ui());
	}

	CUDA_DEV vec3 random_in_unit_sphere() {
		vec3 p;
		do {
			p = 2.0f * vec3(get1f(), get1f(), get1f()) - vec3(1.0f, 1.0f, 1.0f);
		} while (p.squared_length() >= 1.0f);
		return p;
	}
};
