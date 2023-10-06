#pragma once

#include <vector>

template<typename T>
class matrix
{
private:
	std::vector<T> data;
	size_t hg, wd;

public:

	matrix() : data(), hg(0), wd(0) {}

	matrix(
		const size_t h, const size_t w, const T value = (T)0
	) : data(h * w, value), hg(h), wd(w) {}

	matrix(
		const matrix<T> &other
	) : data(other.data), hg(other.hg), wd(other.wd) {}

	matrix(
		matrix<T> &&other
	) : data(std::move(other.data)), hg(other.hg), wd(other.wd) {}

	matrix<T> &operator=(const matrix<T> &other)
	{
		data = other.data;
		hg = other.hg;
		wd = other.wd;
		return *this;
	}

	matrix<T> &operator=(matrix<T> &&other)
	{
		data = std::move(other.data);
		hg = other.hg;
		wd = other.wd;
		return *this;
	}

	void resize(const size_t h, const size_t w, const T value = (T)0)
	{
		hg = h; wd = w;
		data.resize(h * w, value);
	}

	T *operator[](const size_t i)
	{
		return data.data() + i * wd;
	}

	const T *operator[](const size_t i) const
	{
		return data.data() + i * wd;
	}

	size_t height() const { return hg; }
	size_t width() const { return wd; }

	template<typename U>
	friend void swap(matrix<U> &a, matrix<U> &b);
};

template<typename T>
void swap(matrix<T> &a, matrix<T> &b)
{
	std::swap(a.data, b.data);
	size_t t;
	t = a.hg; a.hg = b.hg; b.hg = t;
	t = a.wd; a.wd = b.wd; b.wd = t;
}