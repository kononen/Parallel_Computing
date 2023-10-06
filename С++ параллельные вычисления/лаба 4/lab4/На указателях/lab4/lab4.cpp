#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>
#include <random>
#include "mpi.h"
#include "math.h"

#define _USE_MATH_DEFINES


const double G = 6.67 * 1e-11;
const double eps = 1e-3;


const bool f1 = false; //флаг для чтения тел из файла
const bool f2 = false; //флаг для записи траекторий в текстовый файл

struct body // Структура "тело"
{
	double m;	 // Масса
	double r[3]; // Координаты
	double v[3]; // Скорости

	body operator= (const body& a) {
		(*this).m = a.m;
		for (int k = 0; k < 3; ++k) {
			(*this).r[k] = a.r[k];
			(*this).v[k] = a.v[k];
		}
		return *this;
	}
};




std::ostream& operator<<(std::ostream& str, const body& b)
{
	str << std::setprecision(10) << b.r[0] << " " << b.r[1] << " " << b.r[2] << std::endl;

	return str;
}


inline double norm_vect(const double* r)
{
	return sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
}

inline double max(double a, double b)
{
	return ((a) > (b)) ? (a) : (b);
}

inline double cube(double a)
{
	return a * a * a;
}


void a_calc(double* a, const size_t N, const body* Body_global, const double* r) {

	for (size_t k = 0; k < 3; ++k)
		a[k] = 0.0;

	double diff_r[3] = { 0.0, 0.0, 0.0 };
	double coeff;

	for (size_t j = 0; j < N; ++j)
	{
		for (size_t k = 0; k < 3; ++k)
			diff_r[k] = r[k] - Body_global[j].r[k];

		coeff = cube(1.0 / max(norm_vect(diff_r), eps));

		for (size_t k = 0; k < 3; ++k)
			a[k] -= coeff * G * Body_global[j].m * diff_r[k];
	}

}


void runge_kutta(const size_t N, body* Body_global, body* local_body, const int Nt, const double tau, const double tau_f,
	const int* len, const int* disp, int n_p, int rank, MPI_Datatype MPI_BODY, MPI_Datatype MPI_BODY_R) {

	int n_local = len[rank];

	body* local_body_buf = new body[n_local];

	double a[3] = { 0.0, 0.0, 0.0 }; // Текущие ускорения

	double* a_array = new double[3 * n_local];

	int n_f = round(tau_f / tau); //интервал шагов, через которые происходит запись 

	std::vector<std::ofstream> F;
	if (f2)
	{
		F.resize(n_local); // Массив файлов для каждого узла

		for (size_t i = 0; i < n_local; ++i)
		{
			F[i].open("traj" + std::to_string(disp[rank] + i + 1) + ".txt");
			F[i] << 0.0 << " " << local_body[i] << std::endl;
		}
	}


	for (size_t m = 1; m <= Nt; ++m) {

		for (size_t i = 0; i < n_local; ++i) {

			local_body_buf[i] = local_body[i];

			a_calc(a, N, Body_global, local_body[i].r);

			for (size_t k = 0; k < 3; ++k) {
				(local_body_buf[i].r)[k] += local_body[i].v[k] * tau;
				(local_body_buf[i].v)[k] += a[k] * tau;

				a_array[3 * i + k] = a[k];
			}
		}

		MPI_Allgatherv(local_body_buf, len[rank], MPI_BODY_R, Body_global, len, disp, MPI_BODY_R, MPI_COMM_WORLD);

		for (size_t i = 0; i < n_local; ++i) {

			a_calc(a, N, Body_global, local_body_buf[i].r);

			for (size_t k = 0; k < 3; ++k) {
				local_body[i].r[k] += (local_body_buf[i].v[k] + local_body[i].v[k]) * tau / 2;
				local_body[i].v[k] += (a_array[i * 3 + k] + a[k]) * tau / 2;
			}
		}

		MPI_Allgatherv(local_body, len[rank], MPI_BODY, Body_global, len, disp, MPI_BODY, MPI_COMM_WORLD);


		if (f2 and (m % n_f) == 0)
			for (size_t i = 0; i < n_local; ++i)
				F[i] << tau * m << " " << local_body[i];
	}

	if (f2)
		for (size_t i = 0; i < n_local; ++i)
			F[i].close();

	delete[] a_array;
	delete[]  local_body_buf;
}

void read_file(const std::string& file_name, body* Body, int& N)
{
	std::ifstream f(file_name);


	f >> N;
	Body = new body[N];

	for (int i = 0; i < N; i++)
	{
		f >> Body[i].m;
		for (int k = 0; k < 3; k++) f >> Body[i].r[k];
		for (int k = 0; k < 3; k++) f >> Body[i].v[k];
	}
	f.close();
}

double rand_R(double a, double b) {

	srand(time(NULL));
	return (double)((b - a) * rand()) / RAND_MAX + a;
}

void random_body(body* Body, const int N) {

	Body = new body[N];
	for (size_t i = 0; i < N; ++i) {

		Body[i].m = rand_R(1e5, 1e8);

		for (size_t k = 0; k < 3; ++k) {
			Body[i].r[k] = rand_R(-1e4, 1e4);
			Body[i].v[k] = rand_R(-200, 200);
		}
	}

}

int main(int argc, char** argv)
{
	int rank = 0; // Номер текущего процесса
	int n_p = 0; // Общее число всех процессов 
	int N = 20000;
	double T = 0.02;
	int Nt = 30;

	double tau = T / Nt;
	double tau_f = 0.1; //шаг записи в файл


	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_p);


	int* len = new int[n_p];
	int* displs = new int[n_p];

	const int n = 3; // Количество полей структуры
	int len_struct[3] = { 1, 3, 3 }; // Размеры полей структуры
	MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE }; // Типы полей структуры
	MPI_Aint offsets[3] = { offsetof(body, m), offsetof(body, r), offsetof(body, v) }; // Смещения полей структуры

	MPI_Datatype MPI_BODY;
	MPI_Type_create_struct(n, len_struct, offsets, types, &MPI_BODY);
	MPI_Type_commit(&MPI_BODY);


	// Тип только для позиций
	int n_r = 1;  // Количество элементов структуры
	int len_struct_r[] = { 3 };  // Количество экземпляров
	MPI_Aint offsets_r[] = { offsetof(body, r) }; // Сдвиги относительно начала структуры
	MPI_Datatype types_r[] = { MPI_DOUBLE }; // Типы данных

	MPI_Datatype mpi_tmp; // Вспомогательный тип
	MPI_Type_create_struct(n_r, len_struct_r, offsets_r, types_r, &mpi_tmp);

	MPI_Datatype MPI_BODY_R;
	MPI_Type_create_resized(mpi_tmp, 0, 56, &MPI_BODY_R); // Body состоит из 7 double => 7 x 8 = 56

	MPI_Type_commit(&MPI_BODY_R);

	body* Body_global = new body[N];

	if (rank == 0) {
		if (f1)
			read_file("4body.txt", Body_global, N);
		else
			random_body(Body_global, N);
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // Рассылка числа тел всем процессам
	
	if (rank != 0)
		Body_global = new body[N];

	if (rank == 0)
	{
		int l = N / n_p; // Размер частей массива

		for (int i = 0; i < n_p; ++i) len[i] = l;
		for (int i = 0; i < N % n_p; ++i) ++len[i];

		displs[0] = 0;
		for (int i = 1; i < n_p; ++i) displs[i] = displs[i - 1] + len[i - 1];
	}

	MPI_Bcast(len, n_p, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(displs, n_p, MPI_INT, 0, MPI_COMM_WORLD);

	body* Body_local = new body[len[rank]]; // создание локального массива тел

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Scatterv(Body_global, len, displs, MPI_BODY, Body_local, len[rank], MPI_BODY, 0, MPI_COMM_WORLD); //рассылка локальных тел
	MPI_Bcast(Body_global, N, MPI_BODY, 0, MPI_COMM_WORLD); // Рассылка массива тел всем процессам


	double start, finish;
	double time;


	start = MPI_Wtime();

	runge_kutta(N, Body_global, Body_local, Nt, tau, tau_f, len, displs, n_p, rank, MPI_BODY, MPI_BODY_R);

	finish = MPI_Wtime();

	time = (finish - start) / Nt;

	if (rank == 0) {
		std::cout << "Number of bodies: " << N << std::endl;
		std::cout << "n_p: " << n_p << std::endl;
		std::cout << "Time: " << time << std::endl;
	}

	delete[] len;
	delete[] displs;
	delete[] Body_global;
	delete[] Body_local;
	MPI_Finalize();
	return 0;
}
