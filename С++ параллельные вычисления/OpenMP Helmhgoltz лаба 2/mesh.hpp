#pragma once

#include <cstddef>

#include "vector2d.hpp"

template<typename T>
struct rectangle { T left, bottom, right, top; };

template<typename f_t, typename v_t>
auto project_to_mesh(
	const f_t f,
	const rectangle<v_t> domain,
	const std::size_t nx,
	const std::size_t ny
)
{
	const std::size_t xc = nx + 1, yc = ny + 1;
	const v_t hx = (domain.right - domain.left) / nx;
	const v_t hy = (domain.top - domain.bottom) / ny;

	vector2d<v_t> result(xc, yc);

	#pragma omp parallel for schedule(static)
	for (std::size_t i = 0; i < xc; ++i)
		for (std::size_t j = 0; j < yc; ++j)
			result.sel(i, j) = f(domain.left + i * hx, domain.bottom + j * hy);

	return result;
}