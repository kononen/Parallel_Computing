#pragma once
/// CUDA HOST-SAFE (?)

#include <vector>

#include "vec3.h"

// Bodies are assumed to be material points
struct Body {
	T mass;
	Vec3 position;
	Vec3 velocity; 

public:
	Body() = default;
	Body(T mass, Vec3 x, Vec3 V);

	Body operator=(const Body &other);

	// Conversion
	std::string toString(const std::string &begin = "", const std::string &delimer = " ", const std::string &end = "") const;
};


// ----------------------------------
// --- Utility for N-body problem ---
// ----------------------------------
Vec3 acceleration(const Vec3 &dr, T mass);

void export_time_layer(
	T t,
	const std::vector<Body> &bodies,
	const std::vector<std::string> &filenames,
	int starting_from_body = 0 // allows us to export time layer partially starting at some body
);

/// TEMP
void print_array_of_bodies(int rank, const std::vector<Body> &bodies);
void print_array_of_string(int rank, const std::vector<std::string> &vec);