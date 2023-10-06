#pragma once

#include "body.h"


void nbody_serial(
	std::vector<Body> bodies, // pass-by-copy since we're going to modify bodies localy 
	const T time_interval,
	const size_t iterations,
	const size_t time_layer_export_step,
	const std::string &output_folder,
	const bool benchmark_mode = false
);