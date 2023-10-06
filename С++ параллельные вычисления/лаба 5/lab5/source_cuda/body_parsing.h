#pragma once

#include "body.h"

// Input format: 
// N - number of bodies
// m1 x10 y10 z10 Vx10 Vy10 Vz10 - mass, position, velocity (1-st body)
// ...
// mN xN0 yN0 zN0 VxN0 VyN0 VzN0 - mass, position, velocity (N-th body)
//
std::vector<Body> parse_bodies_from_file(
	const std::string &input_filename
);

void save_bodies_to_file(
	const std::string &output_filename,
	const std::vector<Body> &bodies
);

// Generates N bodies uniformly distributed inside a sphere (or spherical layer)
std::vector<Body> generate_random_input(
	size_t N,
	T m_min, T m_max, // mass (range)
	T r_min, T r_max, // position (spherical layer)
	T v_min, T v_max // velocity (range) (velocities have random directions)
);