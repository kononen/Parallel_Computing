#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <omp.h>

template<typename T>
class vector2d
{
private:
	std::vector<T> data;
	std::size_t hg, wd;

public:
	vector2d() : data(), hg(), wd() {}

	vector2d(
		const std::size_t h, const std::size_t w
	) : data(h * w), hg(h), wd(w) {}

	vector2d(
		const vector2d<T> &other
	) : data(other.data), hg(other.hg), wd(other.wd) {}

	vector2d(
		vector2d<T> &&other
	) : data(std::move(other.data)), hg(other.hg), wd(other.wd) {}

	vector2d<T> &operator=(const vector2d<T> &other)
	{
		data = other.data;
		hg = other.hg;
		wd = other.wd;
		return *this;
	}

	vector2d<T> &operator=(vector2d<T> &&other)
	{
		data = std::move(other.data);
		hg = other.hg;
		wd = other.wd;
		return *this;
	}

	inline std::size_t height() const { return hg; }
	inline std::size_t width() const { return wd; }

	inline T &sel(const std::size_t i, const std::size_t j)
	{
		return data[i * wd + j];
	}

	inline const T& sel(const std::size_t i, const std::size_t j) const
	{
		return data[i * wd + j];
	}

	inline T& safe(const std::size_t i, const std::size_t j)
	{
		return data.at(i * wd + j);
	}

	inline const T& safe(const std::size_t i, const std::size_t j) const
	{
		return data.at(i * wd + j);
	}

	inline vector2d<T> fill_zeros()
	{
		auto it = data.begin();
		const auto end = data.end();
		while (it != end) *it++ = (T)0;
		return *this;
	}

	inline static vector2d<T> zeros(const std::size_t h, const std::size_t w)
	{
		vector2d<T> result(h, w);
		return result.fill_zeros();
	}

	template<typename U>
	friend inline void swap(vector2d<U> &a, vector2d<U> &b);
};

template<typename T>
inline void swap(vector2d<T> &a, vector2d<T> &b)
{
	std::swap(a.data, b.data);
	std::size_t t;
	t = a.hg; a.hg = b.hg; b.hg = t;
	t = a.wd; a.wd = b.wd; b.wd = t;
}

template<typename T>
auto Euclidean_distance(const vector2d<T> &a, const vector2d<T> &b)
{
	const std::size_t hg = a.height(), wd = a.width();
	if (b.height() != hg || b.width() != wd)
		throw std::invalid_argument("Mismatch of dimensions.");

	const std::size_t numthr = omp_get_max_threads();
	std::vector<decltype(std::abs(a.sel(0, 0)))> ds(numthr, 0);

	#pragma omp parallel for schedule(static)
	for (std::size_t i = 0; i < hg; ++i)
	{
		for (std::size_t j = 0; j < wd; ++j)
		{
			const auto d = std::abs(a.sel(i, j) - b.sel(i, j));
			ds[omp_get_thread_num()] += d * d;
		}
	}

	auto sum = ds[0];
	for (std::size_t i = 1; i < numthr; ++i)
		sum += ds[i];

	return std::sqrt(sum);
}

template<typename T>
auto Chebyshev_distance(const vector2d<T> &a, const vector2d<T> &b)
{
	const std::size_t hg = a.height(), wd = a.width();
	if (b.height() != hg || b.width() != wd)
		throw std::invalid_argument("Mismatch of dimensions.");

	const std::size_t numthr = omp_get_max_threads();
	std::vector<decltype(std::abs(a.sel(0, 0)))> ds(numthr, 0);

	#pragma omp parallel for schedule(static)
	for (std::size_t i = 0; i < hg; ++i)
	{
		for (std::size_t j = 0; j < wd; ++j)
		{
			const auto d = std::abs(a.sel(i, j) - b.sel(i, j));
			const std::size_t k = omp_get_thread_num();
			if (ds[k] < d) ds[k] = d;
		}
	}

	auto result = ds[0];
	for (std::size_t i = 1; i < numthr; ++i)
		if (result < ds[i]) result = ds[i];

	return result;
}