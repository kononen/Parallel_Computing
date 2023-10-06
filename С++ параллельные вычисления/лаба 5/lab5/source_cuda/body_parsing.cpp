#include "body_parsing.h"

#include <fstream>

#include "utils_misc.hpp"

// Input format: 
// N - number of bodies
// m1 x10 y10 z10 Vx10 Vy10 Vz10 - mass, position, velocity (1-st body)
// ...
// mN xN0 yN0 zN0 VxN0 VyN0 VzN0 - mass, position, velocity (N-th body)
//
std::vector<Body> parse_bodies_from_file(
	const std::string &input_filename
) {
	std::ifstream inFile(input_filename);
	if (!inFile.is_open()) exit_with_error("Could not open input file");

	size_t N;
	inFile >> N;

	std::vector<Body> bodies(N);

	for (size_t i = 0; i < N; ++i)
		inFile
		>> bodies[i].mass
		>> bodies[i].position.x >> bodies[i].position.y >> bodies[i].position.z
		>> bodies[i].velocity.x >> bodies[i].velocity.y >> bodies[i].velocity.z;

	return bodies;
}

void save_bodies_to_file(
	const std::string &output_filename,
	const std::vector<Body> &bodies
) {
	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) exit_with_error("Could not open file " + output_filename);

	outFile
		<< bodies.size() << '\n';

	for (size_t i = 0; i < bodies.size(); ++i)
		outFile
		<< bodies[i].mass << ' '
		<< bodies[i].position.x << ' ' << bodies[i].position.y << ' ' << bodies[i].position.z << ' '
		<< bodies[i].velocity.x << ' ' << bodies[i].velocity.y << ' ' << bodies[i].velocity.z << '\n';
}

// Generates N bodies uniformly distributed inside a sphere (or spherical layer)
std::vector<Body> generate_random_input(
	size_t N,
	T m_min, T m_max, // mass (range)
	T r_min, T r_max, // position (spherical layer)
	T v_min, T v_max // velocity (range) (velocities have random directions)
) {
	srand(1);

	std::vector<Body> bodies;
	bodies.reserve(N);

	for (size_t i = 0; i < N; ++i) {
		const T mass = rand_T(m_min, m_max);
		const Vec3 position = make_random_vector(r_min, r_max);
		const Vec3 velocity = make_random_vector(v_min, v_max);

		bodies.emplace_back(mass, position, velocity);
	}

	return bodies;
}