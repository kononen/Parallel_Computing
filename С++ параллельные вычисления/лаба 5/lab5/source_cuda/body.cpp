#include "body.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#include "utils_misc.hpp"


// ------------------
// --- Class Body ---
// ------------------
Body::Body(T mass, Vec3 x, Vec3 V) : mass(mass), position(x), velocity(V) {};

Body Body::operator=(const Body & other) {
	this->position = other.position; this->velocity = other.velocity; this->mass = other.mass;
	return *this;
}

// Conversion
std::string Body::toString(const std::string &begin, const std::string &delimer, const std::string &end) const {
	return begin +
		std::to_string(this->mass) + delimer +
		position.toString() + delimer +
		velocity.toString() +
		end;
}


// ----------------------------------
// --- Utility for N-body problem ---
// ----------------------------------
Vec3 acceleration(const Vec3 &dr, T mass) { // does not multiply by G so we can multiply by it extarnally
	const T EPSILON = 1e-8;
	const T dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + EPSILON;
	return dr * (mass / sqrt(dr2));
}

void export_time_layer(
	T t,
	const std::vector<Body> &bodies,
	const std::vector<std::string> &filenames,
	int starting_from_body // allows us to export time layer partially starting at some body
) {
	const size_t N = filenames.size();

	std::ofstream outFile;

	const std::streamsize PRECISION = 17;

	/// Saving to a single memory-mapped file would be orders of magnitude faster, however we trade speed
	/// for debugging convenience and save positions to separate files for each body
	for (size_t i = 0; i < N; ++i) {

		outFile.open(filenames[i], std::ios::app); // open for append
		if (!outFile.is_open()) exit_with_error("Could not open output file " + filenames[i]);
		// we have to reopen and close files due to limitation on simultanious file handles 
		// (~a few hundred), otherwise keeping them open would be faster

		outFile
			<< std::setprecision(PRECISION) << t << ' '
			<< std::setprecision(PRECISION) << bodies[starting_from_body + i].position.x << ' '
			<< std::setprecision(PRECISION) << bodies[starting_from_body + i].position.y << ' '
			<< std::setprecision(PRECISION) << bodies[starting_from_body + i].position.z << '\n';

		outFile.close();
	}
}

/// TEMP
void print_array_of_bodies(int rank, const std::vector<Body> &bodies) {

	std::string str;

	str.append("-> Rank {");
	str.append(std::to_string(rank));
	str.append("}:\n");

	for (int i = 0; i < bodies.size(); ++i) {
		const auto &body = bodies[i];

		str.append("\nmass=[");
		str.append(std::to_string(body.mass));
		str.append("]");

		str.append("\npos={");
		str.append(std::to_string(body.position.x));
		str.append(", ");

		str.append("");
		str.append(std::to_string(body.position.y));
		str.append(", ");

		str.append("");
		str.append(std::to_string(body.position.z));
		str.append("}");

		str.append("\nvel={");
		str.append(std::to_string(body.velocity.x));
		str.append(", ");

		str.append("");
		str.append(std::to_string(body.velocity.y));
		str.append(", ");

		str.append("");
		str.append(std::to_string(body.velocity.z));
		str.append("}\n");
	}

	std::cout << str << std::endl;
}

void print_array_of_string(int rank, const std::vector<std::string> &vec) {

	std::string str;

	str.append("-> Rank {");
	str.append(std::to_string(rank));
	str.append("}: ");

	for (int i = 0; i < vec.size(); ++i) {

		str.append("(");
		str.append(vec[i]);
		str.append(")");
	}

	std::cout << str << std::endl;
}
