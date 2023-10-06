#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iomanip>
#include <omp.h>

#define MyType float

const int blockS = 512;

struct Body // Структура "тело"
{
    MyType m;    // Масса
    MyType r[3]; // Координаты
    MyType v[3]; // Скорости

    // Перегрузка оператора присваивания
    __device__ __host__ Body& operator=(const Body& p)
    {
        m = p.m;
        for (int i = 0; i < 3; ++i)
        {
            r[i] = p.r[i];
            v[i] = p.v[i];
        }
        return *this;
    }
};

// Перегрузка оператора вывода для структуры "тело"
std::ostream& operator<<(std::ostream& str, const Body& b)
{
    str << std::setprecision(10) << b.r[0] << " " << b.r[1] << " " << b.r[2] << std::endl;

    return str;
}

void WriteFile(const std::string& file, const MyType r[3], MyType t, int glob_i)
{
    std::ofstream F(file + std::to_string(glob_i) + ".txt", std::ios::app);
    F << std::setprecision(10) << t << " " << r[0] << " " << r[1] << " " << r[2] << std::endl;
    F.close();
    F.clear();
}

// Модуль вектора
__device__ inline MyType My_norm_vec(const MyType* r)
{
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
}

// Вычисление ускорения
__device__ void My_a(MyType* a, const size_t N, const MyType* r, const int glob_i, const Body* data, MyType G)
{
    MyType buf[3] = { 0.0, 0.0, 0.0 };

    for (size_t k = 0; k < 3; ++k)
        a[k] = 0.0;

    MyType coefN = 1.0;
arr
    Body bod_j;

    float4 dob4;

    __shared__ float4 SharedBlock[512];

    for (int k = 0; k < N / blockDim.x; ++k)
    {
        bod_j = data[blockDim.x * k + threadIdx.x];

        SharedBlock[threadIdx.x] = make_float4(bod_j.r[0], bod_j.r[1], bod_j.r[2], bod_j.m);

        __syncthreads();

        for (size_t j = 0; j < blockDim.x; ++j)
        {
            if (glob_i == blockDim.x * k + j)
                continue;

            dob4 = SharedBlock[j];

            // for (size_t k = 0; k < 3; ++k)
                // buf[k] = bod_j[k] - r[k];

            buf[0] = dob4.x - r[0];
            buf[1] = dob4.y - r[1];
            buf[2] = dob4.z - r[2];

            coefN = buf[0] * buf[0] + buf[1] * buf[1] + buf[2] * buf[2]; //My_norm_vec(buf);

            //coefN *= sqrtf(coefN);

            // coefN *= coefN * coefN;

            coefN = __fdividef(rsqrtf(coefN), coefN) * dob4.w;

#pragma unroll
            for (size_t k = 0; k < 3; ++k)
            {
                a[k] += coefN * buf[k];
            }

        }

        __syncthreads();
    }

#pragma unroll
    for (size_t k = 0; k < 3; ++k)
    {
        a[k] *= G;
    }
}

__global__ void simulate(int N, Body* data, MyType tau, int flagF)
{
    MyType timeStep = 1e-1;  // Шаг записи в файлы

    MyType tn = 1.0; // Конечный момент времени

    size_t Nt = round(tn / tau);       // Количество шагов по времени
    size_t tf = round(timeStep / tau); // Коэффициент пропорциональности шагов

    const MyType G = 6.67e-11; // Гравитационная постоянная

    int glob_i = blockIdx.x * blockDim.x + threadIdx.x; // Текущий номер

    // std::ofstream F("Body_" + std::to_string(glob_i) + ".txt"); // Файл выходных данных
    // std::ofstream F("Body.txt", std::ios::app); // Файл выходных данных

    Body bod_i = data[glob_i]; // Текущее тело

    // F << 0.0 << " " << bod_i;

    MyType a[3] = { 0.0, 0.0, 0.0 }; // Текущие ускорения
    MyType w[3] = { 0.0, 0.0, 0.0 }; // Начальные ускорения

    Body buf;

    // Body* globData = new Body[N];

    // Расчётная схема

    if (glob_i < N)

        for (size_t t = 1; t <= Nt; ++t)
        {
            buf = bod_i;
            My_a(w, N, bod_i.r, glob_i, data, G);

            for (size_t k = 0; k < 3; ++k)
                buf.r[k] += tau * bod_i.v[k];

            data[glob_i] = buf;

            __syncthreads();

            My_a(a, N, buf.r, glob_i, data, G);

            for (size_t k = 0; k < 3; ++k)
            {
                bod_i.r[k] += tau * (bod_i.v[k] + 0.5 * tau * w[k]);
                bod_i.v[k] += 0.5 * tau * (w[k] + a[k]);
            }

            data[glob_i] = bod_i;

            __syncthreads();

            // if (t % tf == 0 && flagF)
                // F << t * tau << " " << bod_i;
        }

    // F.close();
}

int main(int argc, char** argv)
{
    size_t block = 512; // Размер блока

    size_t N = 100000; // Количество тел

    MyType tau = 1e-1; // Шаг по времени

    int flagInitData = 0; // != 0 - считывать из файла, 0 - заполнять случайно
    int flagF = 0;        // != 0 - записывать в файлы, 0 - нет

    MyType mL = 1e+9;  // Нижняя и верхняя  
    MyType mR = 1e+10; // границы значения масс

    MyType rL = -1.0;  // Нижняя и верхняя
    MyType rR = 1.0;   // границы значения координат

    MyType vL = -1.0;  // Нижняя и верхняя
    MyType vR = 1.0;   // границы значения скоростей

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<MyType> dis(0.0, 1.0); // Случайные числа от 0 до 1


    Body* data; // Массив "тел"

    if (flagInitData)
    {
        std::ifstream F("Input.txt"); // Файл входных данных

        F >> N; // Количество тел

        data = new Body[N]; // Массив "тел"

        for (size_t i = 0; i < N; ++i)
            F >> data[i].m >> data[i].r[0] >> data[i].r[1] >> data[i].r[2] >> data[i].v[0] >> data[i].v[1] >> data[i].v[2];

        F.close();
    }
    else
    {
        data = new Body[N]; // Массив "тел"

        for (size_t i = 0; i < N; ++i)
        {
            data[i].m = mL + dis(gen) * (mR - mL);

            for (size_t k = 0; k < 3; ++k)
            {
                data[i].r[k] = rL + dis(gen) * (rR - rL);
                data[i].v[k] = vL + dis(gen) * (vR - vL);
            }
        }
    }

    std::cout << "N = " << N << std::endl;
    std::cout << "tau = " << tau << std::endl;

    Body* GPUdata;

    // Копирование массива data в видеокарты
    cudaMalloc((void**)&GPUdata, N * sizeof(Body));
    cudaMemcpy(GPUdata, data, N * sizeof(Body), cudaMemcpyHostToDevice);

    // Обработчик событий 
    cudaEvent_t start, stop;
    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    size_t blockСount = 0;

    if (N < block)
    {
        simulate << <1, N >> > (N, GPUdata, tau, flagF);
    }
    else
    {
        blockСount = N / block + N % block;
        simulate << <blockСount, block >> > (N, GPUdata, tau, flagF);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "time = " << time / 1000.0 << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(data, GPUdata, N * sizeof(Body), cudaMemcpyHostToDevice);

    cudaFree(GPUdata);
    delete[]data;

    return 0;
}