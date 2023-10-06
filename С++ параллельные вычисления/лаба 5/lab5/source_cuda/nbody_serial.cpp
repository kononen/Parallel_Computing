#include "nbody_serial.h"

#include "utils_file_system.h"

void nbody_serial(
	std::vector<Body> bodies, // pas-by-copy since we're going to modify bodies localy 
	const T time_interval,
	const size_t iterations,
	const size_t time_layer_export_step,
	const std::string &output_folder,
	const bool benchmark_mode
) {
	const size_t N = bodies.size();
	const T tau = time_interval / iterations;
	const T halftau = tau / 2;

	std::vector<std::string> filenames;
	if (!benchmark_mode) {
		// Each body stores it's positions in a separate file 'positions_<index>'
		// using 't  pos.x  pos.y  pos.z' format on each line

		// Setup filenames
		filenames.resize(N);

		const int max_number_width = 1 + static_cast<int>(log10(N));

		for (size_t i = 0; i < N; ++i) {
			const int number_width = 1 + static_cast<int>(log10(i + 1));

			filenames[i] =
				output_folder + "/positions_" +
				std::string(max_number_width - number_width, '0') + std::to_string(i + 1) + ".txt";
		}

		// Reset files and export first time layer
		ensure_folder(output_folder);
		clean_folder(output_folder);
		create_empty_files(filenames);
		export_time_layer(T(0), bodies, filenames);
	}

	// Preallocate space for Runge-Kutta temp variables
	std::vector<Vec3> a(N);
	Vec3 temp_a;
	std::vector<Body> temp_bodies(N);

	// ---------------------------
	// --- Iteration over time ---
	// ---------------------------
	for (size_t step = 0; step < iterations; step++) {
		// ---------------------
		// --- Runge-Kutta 2 ---
		// ---------------------

		std::copy(bodies.begin(), bodies.end(), temp_bodies.begin());

		// k1 = f(t_n, y_n)
		// k2 = f(t_n + tau, y_n + tau * k1 )
		for (size_t i = 0; i < N; ++i) {
			temp_a.set_zero();
			const Vec3& posI = bodies[i].position;
			for (size_t j = 0; j < N; ++j) if (i != j)
				temp_a -= acceleration(posI - bodies[j].position, bodies[j].mass);

			a[i] = temp_a;

			temp_bodies[i].position += bodies[i].velocity * tau;
			temp_bodies[i].velocity += temp_a * (tau * G);
		}
		// y_n+1 = y_n + tau * k2
		for (size_t i = 0; i < N; ++i) {
			temp_a.set_zero();
			const Vec3& tposI = temp_bodies[i].position;
			for (size_t j = 0; j < N; ++j) if (i != j)
				temp_a -= acceleration(tposI - temp_bodies[j].position, bodies[j].mass);

			bodies[i].position += (bodies[i].velocity + temp_bodies[i].velocity) * halftau;
			bodies[i].velocity += (a[i] + temp_a) * (G * halftau);
		}

		// Export time layer with respect to 'TIME_LAYER_EXPORT_STEP'
		if (!benchmark_mode && !((step + 1) % time_layer_export_step))
			export_time_layer(tau * (step + 1), bodies, filenames);
	}


}

// Explicit Euler iteration
		/*for (size_t i = 0; i < N; ++i) {
			Vec3 accel = { 0, 0, 0 };
			for (size_t j = 0; j < N; ++j) {
				const auto dr = bodies[i].position - bodies[j].position;
				accel -= acceleration(dr, bodies[j].mass, 1e-3);
			}
			bodies[i].position += bodies[i].velocity * tau;
			bodies[i].velocity += accel * tau;
		}*/