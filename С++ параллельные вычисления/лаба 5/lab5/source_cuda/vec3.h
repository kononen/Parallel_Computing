#pragma once
/// CUDA HOST-SAFE (?)

#include <string>

#include "utils_math.hpp"

struct Vec3 {
	T x;
	T y;
	T z;
	T w;

public:
	Vec3() = default;
	Vec3(T x, T y, T z);
	Vec3(const Vec3 &other);

	Vec3 operator=(const Vec3 &other);

	// Arithmetic
	Vec3 operator+(const Vec3 &other) const;
	Vec3 operator-(const Vec3 &other) const;
	Vec3 operator*(T val) const;
	Vec3 operator/(T val) const;

	// Increments
	Vec3& operator+=(const Vec3 &other);
	Vec3& operator-=(const Vec3 &other);
	Vec3& operator*=(T value);
	Vec3& operator/=(T value);

	// Conversion
	std::string toString(const std::string &begin = "", const std::string &delimer = " ", const std::string &end = "") const;

	T norm() const;
	T norm2() const;
	T norm3() const;

	void set(T x, T y, T z);
	void set_zero();
};

Vec3 make_random_vector(T length_min, T length_max);