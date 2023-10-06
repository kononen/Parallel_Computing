#pragma once

#include <stdexcept>
#include <sstream>
#include <cmath>

#include "vector2d.hpp"
#include "mesh.hpp"

template<typename k_t, typename f_t>
struct Helmholtz_equation_2d { k_t k; f_t f; };

namespace process_info
{
	enum type_toggle { NONE, ERROR, ITERATIONS_COUNT, ALL };

	struct none
	{
		static constexpr type_toggle type = NONE;

		template<typename U, typename V>
		none &set_results(const U er, const V ic)
		{
			return *this;
		}
	};

	template<typename T, typename U>
	struct error
	{
		static constexpr type_toggle type = ERROR;

		T exact_solution;
		U *err;

		error() : exact_solution(nullptr), err(nullptr) {}

		error(const T es, U *const er) : exact_solution(es), err(er) {}

		template<typename V>
		error<T, U> &set_results(const U er, const V ic)
		{
			*err = er;
			return *this;
		}
	};

	template<typename V>
	struct iterations_count
	{
		static constexpr type_toggle type = ITERATIONS_COUNT;

		V *iter_count;

		iterations_count(V *const ic = nullptr) : iter_count(ic) {}

		template<typename U>
		iterations_count<V> &set_results(const U er, const V ic)
		{
			*iter_count = ic;
			return *this;
		}
	};

	template<typename T, typename U, typename V>
	struct all
	{
		static constexpr type_toggle type = ALL;

		T exact_solution;
		U *err;
		V *iter_count;

		all() : exact_solution(nullptr), err(nullptr), iter_count(nullptr) {}

		all(
			const T es, U *const er, V *const ic
		) : exact_solution(es), err(er), iter_count(ic) {}

		all<T, U, V> &set_results(const U er, const V ic)
		{
			*err = er;
			*iter_count = ic;
			return *this;
		}
	};
}

const double EPSILON = 1e-10;

template<typename T, typename f_t, typename dist_t, typename eps_t, typename info_t = process_info::none>
vector2d<T> Jacobi_method(
	const Helmholtz_equation_2d<T, f_t> &eq,
	const rectangle<T> domain,
	vector2d<T> curr,
	const dist_t dist,
	const eps_t epsilon,
	info_t info = info_t()
)
{
	const std::size_t xc = curr.height(), yc = curr.width();
	const std::size_t nx = xc - 1, ny = yc - 1;
	const auto hx = (domain.right - domain.left) / nx;
	const auto hy = (domain.top - domain.bottom) / ny;

	if (std::abs(hx - hy) >= EPSILON)
	{
		std::ostringstream sout;
		sout
			<< "(|h_x - h_y| == " << std::abs(hx - hy) << ") received while "
			<< "(|h_x - h_y| < " << EPSILON << ") required where "
			<< "h_x = " << hx << ", h_y = " << hy;
		throw std::invalid_argument(sout.str());
	}

	const auto h = hx;
	const auto h2 = h * h;
	const auto k2 = eq.k * eq.k;
	const auto P = 1 / (4 + k2 * h2);
	const auto Q = h2 * P;

	vector2d<T> prev(xc, yc);
	unsigned long long int iter_count = 0;
	do
	{
		swap(prev, curr);

		#pragma omp parallel for schedule(static)
		for (std::size_t i = 1; i < nx; ++i)
		{
			for (std::size_t j = 1; j < ny; ++j)
			{
				curr.sel(i, j) =
					Q * eq.f(domain.left + i * h, domain.bottom + j * h) +
					P * (
						prev.sel(i + 1, j) +
						prev.sel(i - 1, j) +
						prev.sel(i, j + 1) +
						prev.sel(i, j - 1)
					);
			}
		}

		++iter_count;
	}
	while (dist(curr, prev) >= epsilon);

	vector2d<T> exact(xc, yc);

	if constexpr (info_t::type == process_info::ERROR || info_t::type == process_info::ALL)
		exact = project_to_mesh(info.exact_solution, domain, nx, ny);

	info.set_results(dist(curr, exact), iter_count);

	return curr;
}

template<typename T, typename f_t, typename dist_t, typename eps_t, typename info_t = process_info::none>
vector2d<T> Seidel_method(
	const Helmholtz_equation_2d<T, f_t> &eq,
	const rectangle<T> domain,
	vector2d<T> curr,
	const dist_t dist,
	const eps_t epsilon,
	info_t info = info_t()
)
{
	const std::size_t xc = curr.height(), yc = curr.width();
	const std::size_t nx = xc - 1, ny = yc - 1;
	const auto hx = (domain.right - domain.left) / nx;
	const auto hy = (domain.top - domain.bottom) / ny;

	if (std::abs(hx - hy) >= EPSILON)
	{
		std::ostringstream sout;
		sout
			<< "(|h_x - h_y| == " << std::abs(hx - hy) << ") received while "
			<< "(|h_x - h_y| < " << EPSILON << ") required where "
			<< "h_x = " << hx << ", h_y = " << hy;
		throw std::invalid_argument(sout.str());
	}

	const auto h = hx;
	const auto h2 = h * h;
	const auto k2 = eq.k * eq.k;
	const auto P = 1 / (4 + k2 * h2);
	const auto Q = h2 * P;

	vector2d<T> prev(xc, yc);
	unsigned long long int iter_count = 0;
	do
	{
		swap(prev, curr);

		#pragma omp parallel
		{
			#pragma omp for schedule(static)
			for (std::size_t i = 1; i < nx; ++i)
			{
				for (std::size_t j = 1 + (i & 1); j < ny; j += 2)
				{
					curr.sel(i, j) =
						Q * eq.f(domain.left + i * h, domain.bottom + j * h) +
						P * (
							prev.sel(i + 1, j) +
							prev.sel(i - 1, j) +
							prev.sel(i, j + 1) +
							prev.sel(i, j - 1)
						);
				}
			}
			#pragma omp barrier
			#pragma omp for schedule(static)
			for (std::size_t i = 1; i < nx; ++i)
			{
				for (std::size_t j = 2 - (i & 1); j < ny; j += 2)
				{
					curr.sel(i, j) =
						Q * eq.f(domain.left + i * h, domain.bottom + j * h) +
						P * (
							curr.sel(i + 1, j) +
							curr.sel(i - 1, j) +
							curr.sel(i, j + 1) +
							curr.sel(i, j - 1)
						);
				}
			}
		}

		++iter_count;
	}
	while (dist(curr, prev) >= epsilon);

	vector2d<T> exact(xc, yc);

	if constexpr (info_t::type == process_info::ERROR || info_t::type == process_info::ALL)
		exact = project_to_mesh(info.exact_solution, domain, nx, ny);

	info.set_results(dist(curr, exact), iter_count);

	return curr;
}