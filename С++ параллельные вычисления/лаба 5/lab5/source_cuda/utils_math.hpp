#pragma once
/// CUDA KERNEL-SAFE

#include <cmath>

#include "CONFIG.hpp"


// ------------
// --- Math ---
// ------------
const T PI = 3.14159265358979323846;

const T G = 6.67e-11;
const T DR_EPSILON = 1e-10; // used in Newton's law of universal gravitation to avoid division by 0

template<typename _Ty>
constexpr _Ty sqr(_Ty value) { return value * value; } // screw you C++, I want my sqr()

template<typename _Ty>
constexpr _Ty cube(_Ty value) { return value * value * value; } // screw you C++, I want my cube()


// --------------
// --- Random ---
// --------------
inline bool rand_bool() {
	return static_cast<bool>(rand() % 2);
}

inline int rand_int(int min, int max) {
	return min + rand() % (max - min + 1);
}

inline T rand_T() {
	return rand() / (RAND_MAX + 1.);
}

inline T rand_T(T min, T max) {
	return min + (max - min) * rand_T();
}