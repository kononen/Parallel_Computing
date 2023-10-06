#pragma once

#include <iostream>

template<typename T>
struct scientific_form
{
	const T& value;
	scientific_form(const T v) : value(v) {}
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const scientific_form<T> sf)
{
	const auto oldf = out.flags();
	out.setf(std::ios::scientific);
	out << sf.value;
	out.flags(oldf);
	return out;
}