#include <iostream>

#include "vector2d.hpp"
#include "mesh.hpp"
#include "Helmholtz_equation.hpp"
#include "scientific_form.hpp"
#include "timing.hpp"

template<typename T> const T PI_tmpl;

template<> const float
	PI_tmpl<float> = 3.141593F;
template<> const double
	PI_tmpl<double> = 3.1415926535897932;
template<> const long double
	PI_tmpl<long double> = 3.14159265358979323846264338327950288L;

using number = double;
using function = number(*)(const number, const number);

const number PI = PI_tmpl<number>;
const number K = 300;

number u(const number x, const number y)
{
	return (1 - x) * x * std::sin(PI * y);
}

number f(const number x, const number y)
{
	return ((PI * PI + K * K) * (1 - x) * x + 2) * std::sin(PI * y);
}

int main(int argc, char **argv)
{
	try
	{
		const Helmholtz_equation_2d<number, function> equation = { K, f };

		const rectangle<number> domain = { 0, 0, 1, 1 };

		const std::size_t nx = 1000, ny = 1000;

		const auto initial = vector2d<number>::zeros(nx + 1, ny + 1);

		const double epsilon = 1e-6;

		number error;
		unsigned long long int iter_count;

		std::cout
			<< "begin (params)" << std::endl
			<< "\tNx: " << nx << std::endl
			<< "\tNy: " << ny << std::endl
			<< "\tepsilon: " << epsilon << std::endl
			<< "end (params)" << std::endl << std::endl;

		std::chrono::milliseconds t;

		TIMING
			Jacobi_method(
				equation, domain, initial, Chebyshev_distance<number>, epsilon
				//, process_info::error(u, &error)
				//, process_info::iterations_count(&iter_count)
				, process_info::all(u, &error, &iter_count)
			);
		END_TIMING(t)

		std::cout
			<< "begin (Jacobi method)" << std::endl
			<< "\telapsed time: "     << time_formatter(t)      << std::endl
			<< "\titerations count: " << iter_count             << std::endl
			<< "\terror: "            << scientific_form(error) << std::endl
			<< "end (Jacobi method)" << std::endl << std::endl;

		TIMING
			Seidel_method(
				equation, domain, initial, Chebyshev_distance<number>, epsilon
				//, process_info::error(u, &error)
				//, process_info::iterations_count(&iter_count)
				, process_info::all(u, &error, &iter_count)
			);
		END_TIMING(t)

		std::cout
			<< "begin (Seidel method)" << std::endl
			<< "\telapsed time: "     << time_formatter(t)      << std::endl
			<< "\titerations count: " << iter_count             << std::endl
			<< "\terror: "            << scientific_form(error) << std::endl
			<< "end (Seidel method)" << std::endl << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cerr << "[EXCEPTION] " << e.what() << std::endl;
	}

	return 0;
}