#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>

#define TIMING \
{ \
	auto __begin = std::chrono::steady_clock::now();

#define END_TIMING(duration) \
	auto __end = std::chrono::steady_clock::now(); \
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(__end - __begin); \
}

#define OUTPUT_TIME(out, duration) \
if (duration.count() < 1000) \
	out << duration.count() << "ms"; \
else \
{ \
	auto oldf = out.flags(); \
	auto oldp = out.precision(); \
	out.setf(std::ios::fixed); \
	out.precision(3); \
	out << duration.count() / 1000. << "s"; \
	out.flags(oldf); \
	out.precision(oldp); \
}

#define PRINT_TIME(duration) OUTPUT_TIME(std::cout, duration)

struct time_formatter
{
	const std::chrono::milliseconds t;
	time_formatter(const std::chrono::milliseconds __t) : t(__t) {}
};

std::ostream &operator<<(std::ostream &out, const time_formatter &f)
{
	OUTPUT_TIME(out, f.t)
	return out;
}