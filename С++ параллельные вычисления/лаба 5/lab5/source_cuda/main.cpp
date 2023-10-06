#include <iostream>
#include <iomanip>

#include "nbody_serial.h"
#include "nbody_cuda.h"
#include "body_parsing.h"

#include "utils_misc.hpp"
#include "static_timer.hpp"
#include "table.hpp"


int main() {
	// Print CUDA info
	print_devices();

	StaticTimer::start();
	std::cout << ">>> Parsing input...\n";

	// Parse bodies
	std::vector<Body> bodies = USE_RANDOM_BODIES
		// Generate random bodies if 'USE_RANDOM_BODIES' is selected
		? generate_random_input(
			RANDOM_N,
			RANDOM_M_MIN, RANDOM_M_MAX,
			RANDOM_R_MIN, RANDOM_R_MAX,
			RANDOM_V_MIN, RANDOM_V_MAX
		)
		// Parse bodies otherwise
		: parse_bodies_from_file(INPUT_FILENAME);

	// Display config
	std::cout
		<< "\n"
		<< "Parsed in " << StaticTimer::end() << "sec\n"
		<< "--------------------------------------------------\n"
		<< "Input file    -> " << (USE_RANDOM_BODIES ? "<random bodies are generated>" : INPUT_FILENAME) << "\n"
		<< "Output folder -> " << OUTPUT_FOLDER << "\n"
		<< "Order test    -> " << bool_to_str(TEST_CONVERGENCE_ORDER) << "\n"
		<< "Mode          -> " << (BENCHMARK_MODE ? "BENCHMARK" : "DEFAULT") << "\n"
		<< "Type          -> " << typeid(T).name() << "\n"
		<< "N             =  " << bodies.size() << "\n"
		<< "Time interval =  " << TIME_INTERVAL << "\n"
		<< "Iterations    =  " << ITERATIONS << "\n"
		<< "Step          =  " << TIME_INTERVAL / ITERATIONS << "\n"
		<< "Export step   =  " << TIME_LAYER_EXPORT_STEP << "\n"
		<< "--------------------------------------------------\n" << std::endl;

	// ---------------
	// --- Methods ---
	// ---------------
	table_add_1("Method");
	table_add_2("Time (sec)");
	table_add_3("Speedup");
	table_hline();

	double timeSerial = -1;
	double timeParallel = -1;

	// ------------------------
	// --- 1) Serial method ---
	// ------------------------
	// 1. Method
	if (!SKIP_SERIAL_METHOD) {
		table_add_1("Serial");

		// 2. Time
		StaticTimer::start();
		nbody_serial(bodies, TIME_INTERVAL, ITERATIONS, TIME_LAYER_EXPORT_STEP, OUTPUT_FOLDER, BENCHMARK_MODE);
		timeSerial = StaticTimer::end();
		table_add_2(timeSerial);

		// 3. Speedup
		table_add_3(timeSerial / timeSerial);
	}

	// --------------------------
	// --- 2) Parallel method ---
	// --------------------------
	if (!SKIP_CUDA_METHOD) {
		// 1. Method
		table_add_1("CUDA");

		// 2. Time
		StaticTimer::start();
		nbody_cuda(bodies, TIME_INTERVAL, ITERATIONS, TIME_LAYER_EXPORT_STEP, OUTPUT_FOLDER_CUDA, BENCHMARK_MODE);
		timeParallel = StaticTimer::end();
		table_add_2(timeParallel);

		// 3. Speedup
		table_add_3(timeSerial / timeParallel);
	}	


	// ----------------------------
	// --- Order of convergence ---
	// ----------------------------
	if (TEST_CONVERGENCE_ORDER) {
		nbody_cuda(bodies, TIME_INTERVAL, ITERATIONS, TIME_LAYER_EXPORT_STEP, OUTPUT_FOLDER_ORDER_TEST_1, BENCHMARK_MODE);
		nbody_cuda(bodies, TIME_INTERVAL, q * ITERATIONS, q * TIME_LAYER_EXPORT_STEP, OUTPUT_FOLDER_ORDER_TEST_2, BENCHMARK_MODE);
		nbody_cuda(bodies, TIME_INTERVAL, q * q * ITERATIONS, q * q * TIME_LAYER_EXPORT_STEP, OUTPUT_FOLDER_ORDER_TEST_3, BENCHMARK_MODE);
	}
	
  
	// ---------------------
	// --- Finalize CUDA ---
	// ---------------------
	const int error_code = cuda_finish();
	std::cout << "\nExectution finalized with error code " << error_code;
    return error_code;
}
