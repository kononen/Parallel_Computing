#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "mpi.h"

#include "matrix.hpp"

/* тип указателя на функцию
с двумя параметрами типа const double
и типом возвращаемого значения double */
using function = double(*)(const double, const double);

struct Helmholtz_equation_2d
{
	double k;
	function f;
};

const double PI = 3.1415926535897932;
const double K = 2000;

// точное решение задачи
double u(const double x, const double y)
{
	return (1 - x) * x * std::sin(PI * y);
}

// правая часть
double f(const double x, const double y)
{
	return ((PI * PI + K * K) * (1 - x) * x + 2) * std::sin(PI * y);
}

const Helmholtz_equation_2d eq = { K, f };
const int dim = 4001;
const double eps = 1e-6;

void output_to_file(
	const std::string &file_path,
	const matrix<double> &matr,
	const size_t step_x = 1,
	const size_t step_y = 1
)
{
	std::ofstream fout(file_path);

	if (!fout.is_open())
	{
		std::cout
			<< " Procedure \'output_to_file\': can\'t open file \'"
			<< file_path << "\'." << std::endl;
		return;
	}

	const size_t Nx = matr.height(), Ny = matr.width();
	const double hx = 1. / (Nx - 1), hy = 1. / (Ny - 1);

	for (size_t i = 0; i < Nx; i += step_x)
		for (size_t j = 0; j < Ny; j += step_y)
			fout
				<< i * hx << "\t"
				<< j * hy << "\t"
				<< matr[i][j] << std::endl;

	fout.close();
}

// максимум модуля разности: max_{i, j} | a_{ij} - b_{ij} |
double distance(const matrix<double> &a, const matrix<double> &b)
{
	const size_t Nx = a.height(), Ny = a.width();

	double result = fabs(a[0][0] - b[0][0]);
	for (size_t i = 0; i < Nx; ++i)
	{
		for (size_t j = 0; j < Ny; ++j)
		{
			const double r = fabs(a[i][j] - b[i][j]);
			if (result < r)
				result = r;
		}
	}

	return result;
}

// погрешность приближённого решения: max_{i, j} | a_{ij} - u(i h_x, j h_y) |
double error(const matrix<double> &a)
{
	const size_t Nx = a.height(), Ny = a.width();
	const double hx = 1. / (Nx - 1), hy = 1. / (Ny - 1);

	double result = fabs(a[0][0] - u(0, 0));
	for (size_t i = 0; i < Nx; ++i)
	{
		for (size_t j = 0; j < Ny; ++j)
		{
			const double r = fabs(a[i][j] - u(i * hx, j * hy));
			if (result < r)
				result = r;
		}
	}

	return result;
}

template<typename method_t>
void solve(
	const method_t method,
	const Helmholtz_equation_2d eq,
	const int dim,
	const int process_id, const int processes_count,
	const double eps,
	const std::string &method_name
)
{
	const double t_start = MPI_Wtime();

	matrix<double> sol;
	std::vector<int> rows_counts, entry_rows;
	// rows_counts - количества строк в полосах
	// entry_rows - глобальные индексы начальных (верхних) строк полос (т.е. смещения)

	if (process_id == 0)
	{
		sol.resize(dim, dim);
		rows_counts.resize(processes_count, dim / processes_count);
		entry_rows.resize(processes_count);

		// корректирование значений количеств строк в полосах (учёт остатка от деления)
		for (size_t i = 0; i < dim % processes_count; ++i)
			++rows_counts[i];

		entry_rows[0] = 0;
		for (size_t i = 1; i < processes_count; ++i)
			entry_rows[i] = entry_rows[i - 1] + rows_counts[i - 1];
	}

	int rows_count, entry_row;

	MPI_Scatter(
		rows_counts.data(), 1, MPI_INT,
		&rows_count,        1, MPI_INT,
		0, MPI_COMM_WORLD
	);

	MPI_Scatter(
		entry_rows.data(), 1, MPI_INT,
		&entry_row,        1, MPI_INT,
		0, MPI_COMM_WORLD
	);

	int iter_count;

	// вычисление полосы (т.е. части приближённого решения) процессом
	matrix<double> part = method(
		eq, dim,
		rows_count, entry_row,
		process_id, processes_count,
		eps, iter_count
	);

	std::vector<int> elem_counts, entries;

	if (process_id == 0)
	{
		elem_counts.resize(processes_count);
		entries.resize(processes_count);

		for (size_t i = 0; i < processes_count; ++i)
		{
			elem_counts[i] = rows_counts[i] * dim;
			entries[i] = entry_rows[i] * dim;
		}
	}

	MPI_Gatherv(
		part[0], dim * rows_count,                   MPI_DOUBLE,
		sol[0],  elem_counts.data(), entries.data(), MPI_DOUBLE,
		0, MPI_COMM_WORLD
	);

	const double t_elapsed = MPI_Wtime() - t_start;

	if (process_id == 0)
	{
		//output_to_file(method_name + ".txt", sol, 10, 10);
		std::cout
			<< method_name << ": "
			<< "elapsed time = " << std::fixed << std::setprecision(2) << t_elapsed << "s, "
			<< "error = " << std::defaultfloat << std::setprecision(6) << error(sol) << ", "
			<< "iterations count = " << iter_count << ".\n";//*/
	}
}

matrix<double> Send_Recv_Jacobi_method(
	const Helmholtz_equation_2d eq,
	const int dim,
	const int rows_count, const int entry_row,
	const int process_id, const int processes_count,
	const double eps,
	int &iter_count
)
{
	const double h = 1. / (dim - 1);
	const double c = 1. / (4 + eq.k * eq.k * h * h);
	const double d = h * h * c;

	matrix<double> curr(rows_count, dim), prev(rows_count, dim);
	std::vector<double> from_next(dim), from_prev(dim);

	double dist;
	iter_count = 0;

	do {
		swap(prev, curr);

		if (processes_count > 1)
		{
			if (process_id != processes_count - 1)
				MPI_Send(
					prev[rows_count - 1], dim, MPI_DOUBLE,
					process_id + 1, 0, MPI_COMM_WORLD
				);

			if (process_id != 0)
				MPI_Recv(
					from_prev.data(), dim, MPI_DOUBLE,
					process_id - 1, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);

			if (process_id != 0)
				MPI_Send(
					prev[0], dim, MPI_DOUBLE,
					process_id - 1, 0, MPI_COMM_WORLD
				);

			if (process_id != processes_count - 1)
				MPI_Recv(
					from_next.data(), dim, MPI_DOUBLE,
					process_id + 1, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						prev[i + 1][j] +
						prev[i - 1][j] +
						prev[i][j + 1] +
						prev[i][j - 1]
					);

		if (process_id != 0)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						prev[1][j] +
						from_prev[j] +
						prev[0][j + 1] +
						prev[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						prev[rows_count - 2][j] +
						prev[rows_count - 1][j + 1] +
						prev[rows_count - 1][j - 1]
					);

		++iter_count;

		double process_dist = distance(prev, curr);
		//double process_dist = error(prev);
		MPI_Allreduce(&process_dist, &dist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	while (dist > eps);

	return curr;
}

matrix<double> Sendrecv_Jacobi_method(
	const Helmholtz_equation_2d eq,
	const int dim,
	const int rows_count, const int entry_row,
	const int process_id, const int processes_count,
	const double eps,
	int &iter_count
)
{
	const double h = 1. / (dim - 1);
	const double c = 1. / (4 + eq.k * eq.k * h * h);
	const double d = h * h * c;

	const int prev_count = (process_id != 0 ? dim : 0);
	const int next_count = (process_id != processes_count - 1 ? dim : 0);

	const int prev_id = (process_id != 0 ? process_id - 1 : processes_count - 1);
	const int next_id = (process_id != processes_count - 1 ? process_id + 1 : 0);

	matrix<double> curr(rows_count, dim), prev(rows_count, dim);
	std::vector<double> from_next(dim), from_prev(dim);

	double dist;
	iter_count = 0;

	do {
		swap(prev, curr);

		if(processes_count > 1)
		{
			MPI_Sendrecv(
				prev[rows_count - 1], next_count, MPI_DOUBLE, next_id, 0,
				from_prev.data(),     prev_count, MPI_DOUBLE, prev_id, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);

			MPI_Sendrecv(
				prev[0],          prev_count, MPI_DOUBLE, prev_id, 1,
				from_next.data(), next_count, MPI_DOUBLE, next_id, 1,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						prev[i + 1][j] +
						prev[i - 1][j] +
						prev[i][j + 1] +
						prev[i][j - 1]
					);

		if (process_id != 0)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						prev[1][j] +
						from_prev[j] +
						prev[0][j + 1] +
						prev[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						prev[rows_count - 2][j] +
						prev[rows_count - 1][j + 1] +
						prev[rows_count - 1][j - 1]
					);

		++iter_count;

		double process_dist = distance(prev, curr);
		//double process_dist = error(prev);
		MPI_Allreduce(&process_dist, &dist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	while (dist > eps);

	return curr;
}

matrix<double> Isend_Irecv_Jacobi_method(
	const Helmholtz_equation_2d eq,
	const int dim,
	const int rows_count, const int entry_row,
	const int process_id, const int processes_count,
	const double eps,
	int &iter_count
)
{
	const double h = 1. / (dim - 1);
	const double c = 1. / (4 + eq.k * eq.k * h * h);
	const double d = h * h * c;

	matrix<double> curr(rows_count, dim), prev(rows_count, dim);
	std::vector<double> from_next(dim), from_prev(dim);

	MPI_Request prev_req[2], next_req[2];

	double dist;
	iter_count = 0;

	do {
		swap(prev, curr);

		if (process_id != 0)
		{
			MPI_Isend(
				prev[0], dim, MPI_DOUBLE, process_id - 1, 0,
				MPI_COMM_WORLD, prev_req
			);

			MPI_Irecv(
				from_prev.data(), dim, MPI_DOUBLE, process_id - 1, 0,
				MPI_COMM_WORLD, prev_req + 1
			);
		}

		if (process_id != processes_count - 1)
		{
			MPI_Isend(
				prev[rows_count - 1], dim, MPI_DOUBLE, process_id + 1, 0,
				MPI_COMM_WORLD, next_req
			);

			MPI_Irecv(
				from_next.data(), dim, MPI_DOUBLE, process_id + 1, 0,
				MPI_COMM_WORLD, next_req + 1
			);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						prev[i + 1][j] +
						prev[i - 1][j] +
						prev[i][j + 1] +
						prev[i][j - 1]
					);

		if (process_id != 0)
			MPI_Waitall(2, prev_req, MPI_STATUSES_IGNORE);

		if (process_id != processes_count - 1)
			MPI_Waitall(2, next_req, MPI_STATUSES_IGNORE);

		if (process_id != 0)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						prev[1][j] +
						from_prev[j] +
						prev[0][j + 1] +
						prev[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 1; j < dim - 1; ++j)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						prev[rows_count - 2][j] +
						prev[rows_count - 1][j + 1] +
						prev[rows_count - 1][j - 1]
					);

		++iter_count;

		double process_dist = distance(prev, curr);
		//double process_dist = error(prev);
		MPI_Allreduce(&process_dist, &dist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	while (dist > eps);

	return curr;
}

matrix<double> Send_Recv_Seidel_method(
	const Helmholtz_equation_2d eq,
	const int dim,
	const int rows_count, const int entry_row,
	const int process_id, const int processes_count,
	const double eps,
	int &iter_count
)
{
	const double h = 1. / (dim - 1);
	const double c = 1. / (4 + eq.k * eq.k * h * h);
	const double d = h * h * c;

	matrix<double> curr(rows_count, dim), prev(rows_count, dim);
	std::vector<double> from_next(dim), from_prev(dim);

	double dist;
	iter_count = 0;

	do {
		swap(prev, curr);

		if (processes_count > 1)
		{
			if (process_id != processes_count - 1)
				MPI_Send(
					prev[rows_count - 1], dim, MPI_DOUBLE,
					process_id + 1, 0, MPI_COMM_WORLD
				);

			if (process_id != 0)
				MPI_Recv(
					from_prev.data(), dim, MPI_DOUBLE,
					process_id - 1, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);

			if (process_id != 0)
				MPI_Send(
					prev[0], dim, MPI_DOUBLE,
					process_id - 1, 0, MPI_COMM_WORLD
				);

			if (process_id != processes_count - 1)
				MPI_Recv(
					from_next.data(), dim, MPI_DOUBLE,
					process_id + 1, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 1 + (entry_row + i) % 2; j < dim - 1; j += 2)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						prev[i + 1][j] +
						prev[i - 1][j] +
						prev[i][j + 1] +
						prev[i][j - 1]
					);

		if (process_id != 0)
			for (size_t j = 1 + entry_row % 2; j < dim - 1; j += 2)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						prev[1][j] +
						from_prev[j] +
						prev[0][j + 1] +
						prev[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 1 + (entry_row + rows_count - 1) % 2; j < dim - 1; j += 2)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						prev[rows_count - 2][j] +
						prev[rows_count - 1][j + 1] +
						prev[rows_count - 1][j - 1]
					);

		if (processes_count > 1)
		{
			if (process_id != processes_count - 1)
				MPI_Send(
					curr[rows_count - 1], dim, MPI_DOUBLE,
					process_id + 1, 0, MPI_COMM_WORLD
				);

			if (process_id != 0)
				MPI_Recv(
					from_prev.data(), dim, MPI_DOUBLE,
					process_id - 1, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);

			if (process_id != 0)
				MPI_Send(
					curr[0], dim, MPI_DOUBLE,
					process_id - 1, 0, MPI_COMM_WORLD
				);

			if (process_id != processes_count - 1)
				MPI_Recv(
					from_next.data(), dim, MPI_DOUBLE,
					process_id + 1, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 2 - (entry_row + i) % 2; j < dim - 1; j += 2)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						curr[i + 1][j] +
						curr[i - 1][j] +
						curr[i][j + 1] +
						curr[i][j - 1]
					);

		if (process_id != 0)
			for (size_t j = 2 - entry_row % 2; j < dim - 1; j += 2)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						curr[1][j] +
						from_prev[j] +
						curr[0][j + 1] +
						curr[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 2 - (entry_row + rows_count - 1) % 2; j < dim - 1; j += 2)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						curr[rows_count - 2][j] +
						curr[rows_count - 1][j + 1] +
						curr[rows_count - 1][j - 1]
					);

		++iter_count;

		double process_dist = distance(prev, curr);
		//double process_dist = error(prev);
		MPI_Allreduce(&process_dist, &dist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	while (dist > eps);

	return curr;
}

matrix<double> Sendrecv_Seidel_method(
	const Helmholtz_equation_2d eq,
	const int dim,
	const int rows_count, const int entry_row,
	const int process_id, const int processes_count,
	const double eps,
	int &iter_count
)
{
	const double h = 1. / (dim - 1);
	const double c = 1. / (4 + eq.k * eq.k * h * h);
	const double d = h * h * c;

	const int prev_count = (process_id != 0 ? dim : 0);
	const int next_count = (process_id != processes_count - 1 ? dim : 0);

	const int prev_id = (process_id != 0 ? process_id - 1 : processes_count - 1);
	const int next_id = (process_id != processes_count - 1 ? process_id + 1 : 0);

	matrix<double> curr(rows_count, dim), prev(rows_count, dim);
	std::vector<double> from_next(dim), from_prev(dim);

	double dist;
	iter_count = 0;

	do {
		swap(prev, curr);

		if(processes_count > 1)
		{
			MPI_Sendrecv(
				prev[rows_count - 1], next_count, MPI_DOUBLE, next_id, 0,
				from_prev.data(),     prev_count, MPI_DOUBLE, prev_id, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);

			MPI_Sendrecv(
				prev[0],          prev_count, MPI_DOUBLE, prev_id, 1,
				from_next.data(), next_count, MPI_DOUBLE, next_id, 1,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 1 + (entry_row + i) % 2; j < dim - 1; j += 2)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						prev[i + 1][j] +
						prev[i - 1][j] +
						prev[i][j + 1] +
						prev[i][j - 1]
					);

		if (process_id != 0)
			for (size_t j = 1 + entry_row % 2; j < dim - 1; j += 2)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						prev[1][j] +
						from_prev[j] +
						prev[0][j + 1] +
						prev[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 1 + (entry_row + rows_count - 1) % 2; j < dim - 1; j += 2)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						prev[rows_count - 2][j] +
						prev[rows_count - 1][j + 1] +
						prev[rows_count - 1][j - 1]
					);

		if(processes_count > 1)
		{
			MPI_Sendrecv(
				curr[rows_count - 1], next_count, MPI_DOUBLE, next_id, 0,
				from_prev.data(),     prev_count, MPI_DOUBLE, prev_id, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);

			MPI_Sendrecv(
				curr[0],          prev_count, MPI_DOUBLE, prev_id, 1,
				from_next.data(), next_count, MPI_DOUBLE, next_id, 1,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 2 - (entry_row + i) % 2; j < dim - 1; j += 2)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						curr[i + 1][j] +
						curr[i - 1][j] +
						curr[i][j + 1] +
						curr[i][j - 1]
					);

		if (process_id != 0)
			for (size_t j = 2 - entry_row % 2; j < dim - 1; j += 2)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						curr[1][j] +
						from_prev[j] +
						curr[0][j + 1] +
						curr[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 2 - (entry_row + rows_count - 1) % 2; j < dim - 1; j += 2)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						curr[rows_count - 2][j] +
						curr[rows_count - 1][j + 1] +
						curr[rows_count - 1][j - 1]
					);

		++iter_count;

		double process_dist = distance(prev, curr);
		//double process_dist = error(prev);
		MPI_Allreduce(&process_dist, &dist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	while (dist > eps);

	return curr;
}

matrix<double> Isend_Irecv_Seidel_method(
	const Helmholtz_equation_2d eq,
	const int dim,
	const int rows_count, const int entry_row,
	const int process_id, const int processes_count,
	const double eps,
	int &iter_count
)
{
	const double h = 1. / (dim - 1);
	const double c = 1. / (4 + eq.k * eq.k * h * h);
	const double d = h * h * c;

	matrix<double> curr(rows_count, dim), prev(rows_count, dim);
	std::vector<double> from_next(dim), from_prev(dim);

	MPI_Request prev_req[2], next_req[2];

	double dist;
	iter_count = 0;

	do {
		swap(prev, curr);

		if (process_id != 0)
		{
			MPI_Isend(
				prev[0], dim, MPI_DOUBLE, process_id - 1, 0,
				MPI_COMM_WORLD, prev_req
			);

			MPI_Irecv(
				from_prev.data(), dim, MPI_DOUBLE, process_id - 1, 0,
				MPI_COMM_WORLD, prev_req + 1
			);
		}

		if (process_id != processes_count - 1)
		{
			MPI_Isend(
				prev[rows_count - 1], dim, MPI_DOUBLE, process_id + 1, 0,
				MPI_COMM_WORLD, next_req
			);

			MPI_Irecv(
				from_next.data(), dim, MPI_DOUBLE, process_id + 1, 0,
				MPI_COMM_WORLD, next_req + 1
			);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 1 + (entry_row + i) % 2; j < dim - 1; j += 2)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						prev[i + 1][j] +
						prev[i - 1][j] +
						prev[i][j + 1] +
						prev[i][j - 1]
					);

		if (process_id != 0)
			MPI_Waitall(2, prev_req, MPI_STATUSES_IGNORE);

		if (process_id != processes_count - 1)
			MPI_Waitall(2, next_req, MPI_STATUSES_IGNORE);

		if (process_id != 0)
			for (size_t j = 1 + entry_row % 2; j < dim - 1; j += 2)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						prev[1][j] +
						from_prev[j] +
						prev[0][j + 1] +
						prev[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 1 + (entry_row + rows_count - 1) % 2; j < dim - 1; j += 2)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						prev[rows_count - 2][j] +
						prev[rows_count - 1][j + 1] +
						prev[rows_count - 1][j - 1]
					);

		if (process_id != 0)
		{
			MPI_Isend(
				curr[0], dim, MPI_DOUBLE, process_id - 1, 0,
				MPI_COMM_WORLD, prev_req
			);

			MPI_Irecv(
				from_prev.data(), dim, MPI_DOUBLE, process_id - 1, 0,
				MPI_COMM_WORLD, prev_req + 1
			);
		}

		if (process_id != processes_count - 1)
		{
			MPI_Isend(
				curr[rows_count - 1], dim, MPI_DOUBLE, process_id + 1, 0,
				MPI_COMM_WORLD, next_req
			);

			MPI_Irecv(
				from_next.data(), dim, MPI_DOUBLE, process_id + 1, 0,
				MPI_COMM_WORLD, next_req + 1
			);
		}

		for (size_t i = 1; i < rows_count - 1; ++i)
			for (size_t j = 2 - (entry_row + i) % 2; j < dim - 1; j += 2)
				curr[i][j] =
					d * eq.f((entry_row + i) * h, j * h) +
					c * (
						curr[i + 1][j] +
						curr[i - 1][j] +
						curr[i][j + 1] +
						curr[i][j - 1]
					);

		if (process_id != 0)
			MPI_Waitall(2, prev_req, MPI_STATUSES_IGNORE);

		if (process_id != processes_count - 1)
			MPI_Waitall(2, next_req, MPI_STATUSES_IGNORE);

		if (process_id != 0)
			for (size_t j = 2 - entry_row % 2; j < dim - 1; j += 2)
				curr[0][j] =
					d * eq.f(entry_row * h, j * h) +
					c * (
						curr[1][j] +
						from_prev[j] +
						curr[0][j + 1] +
						curr[0][j - 1]
					);

		if (process_id != processes_count - 1)
			for (size_t j = 2 - (entry_row + rows_count - 1) % 2; j < dim - 1; j += 2)
				curr[rows_count - 1][j] =
					d * eq.f((entry_row + rows_count - 1) * h, j * h) +
					c * (
						from_next[j] +
						curr[rows_count - 2][j] +
						curr[rows_count - 1][j + 1] +
						curr[rows_count - 1][j - 1]
					);

		++iter_count;

		double process_dist = distance(prev, curr);
		//double process_dist = error(prev);
		MPI_Allreduce(&process_dist, &dist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	while (dist > eps);

	return curr;
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int processes_count, process_id;

	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

	solve(  Send_Recv_Jacobi_method, eq, dim, process_id, processes_count, eps, "  Send_Recv_Jacobi_method");
	solve(   Sendrecv_Jacobi_method, eq, dim, process_id, processes_count, eps, "   Sendrecv_Jacobi_method");
	solve(Isend_Irecv_Jacobi_method, eq, dim, process_id, processes_count, eps, "Isend_Irecv_Jacobi_method");
	solve(  Send_Recv_Seidel_method, eq, dim, process_id, processes_count, eps, "  Send_Recv_Seidel_method");
	solve(   Sendrecv_Seidel_method, eq, dim, process_id, processes_count, eps, "   Sendrecv_Seidel_method");
	solve(Isend_Irecv_Seidel_method, eq, dim, process_id, processes_count, eps, "Isend_Irecv_Seidel_method");

	//std::cout << "Process " << process_id << " of " << processes_count << " successfully completed." << std::endl;

	MPI_Finalize();

	return 0;
}