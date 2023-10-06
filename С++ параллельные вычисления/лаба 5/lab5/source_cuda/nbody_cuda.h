#pragma once

#include "body.h"

// Detect devices and print their specs
void print_devices();

// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
int cuda_finish();


void nbody_cuda(
	std::vector<Body> bodies, // pas-by-copy since we're going to modify bodies localy 
	const T time_interval,
	const size_t iterations,
	const size_t time_layer_export_step,
	const std::string &output_folder,
	const bool benchmark_mode = false);