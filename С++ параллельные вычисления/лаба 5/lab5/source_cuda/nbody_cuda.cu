#include "nbody_cuda.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "utils_file_system.h"



__device__ inline T4 cuda_get_acceleration(const T4& position_i, const T4& position_j) {
    // dr = position_j - position_i
    T4 dr;
    dr.x = position_i.x - position_j.x;
    dr.y = position_i.y - position_j.y;
    dr.z = position_i.z - position_j.z;

    // dr2 = ||dr||^2 + EPS^2
    const T EPSILON = 1e-8;
    const T dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + EPSILON;

    // dr3inverse = 1/dr2^(3/2)
    #ifdef DATATYPE_FLOAT
    const T mass_dr3inverse = __fdividef(position_j.w, dr2) * __frsqrt_rn(dr2);
    #endif
    #ifdef DATATYPE_DOUBLE
    const T mass_dr3inverse = __ddiv_rn(position_j.w, dr2) * rsqrt(dr2);
    #endif

    // const T mass_dr3inverse = G * position_j.w / dr2 / sqrt(dr2);
    

    // acceleration = m_j * dr3inverse
    T4 acceleration;
    acceleration.x = dr.x * mass_dr3inverse;
    acceleration.y = dr.y * mass_dr3inverse;
    acceleration.z = dr.z * mass_dr3inverse;

    return acceleration;
}

__global__ void cuda_integrate_bodies(Body *bodies, T4* temp_positions, int N, T tau) {
    // 1 thread per body => index corresponds to body
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    T halftau = T(0.5) * tau;

    const int block_size = CUDA_BLOCK_SIZE;

    __shared__ T4 arr[block_size];
    
    T4 temp_accel, accel;
    T4 temp_velocity;
    T4 temp;

    // ---------------------
    // --- Runge-Kutta 2 ---
    // ---------------------

    // k1 = f(t_n, y_n)
    // k2 = f(t_n + tau, y_n + tau * k1 )

    T4 bi;
    bi.x = bodies[i].position.x;
    bi.y = bodies[i].position.y;
    bi.z = bodies[i].position.z;
    bi.w = bodies[i].mass;

    // Compute accelerations
    temp_accel.x = 0;
    temp_accel.y = 0;
    temp_accel.z = 0;

    for (int block = 0; block < N / block_size; ++block) {
        
        const int offset = block * block_size;

        arr[threadIdx.x].x = bodies[offset + threadIdx.x].position.x;
        arr[threadIdx.x].y = bodies[offset + threadIdx.x].position.y;
        arr[threadIdx.x].z = bodies[offset + threadIdx.x].position.z;
        arr[threadIdx.x].w = bodies[offset + threadIdx.x].mass;

        __syncthreads();

        for (size_t j = 0; j <  block_size; ++j) {
            temp = cuda_get_acceleration(bi, arr[j]);
        
            // acceleration
            temp_accel.x -= temp.x;
            temp_accel.y -= temp.y;
            temp_accel.z -= temp.z;
        }

        __syncthreads();
    }

    // Set temp positions, velocities
    temp_positions[i].x = bodies[i].position.x + bodies[i].velocity.x * tau;
    temp_positions[i].y = bodies[i].position.y + bodies[i].velocity.y * tau;
    temp_positions[i].z = bodies[i].position.z + bodies[i].velocity.z * tau;
    temp_positions[i].w = bodies[i].mass;

    temp_velocity.x = bodies[i].velocity.x + temp_accel.x * tau;
    temp_velocity.y = bodies[i].velocity.y + temp_accel.y * tau;
    temp_velocity.z = bodies[i].velocity.z + temp_accel.z * tau;

    // y_n+1 = y_n + tau * k2

    // Compute accelerations
    const T4 temp_bi = temp_positions[i];

    accel.x = 0;
    accel.y = 0;
    accel.z = 0;

    for (int block = 0; block < N / block_size; ++block) {

        const int offset = block * block_size;

        arr[threadIdx.x] = temp_positions[offset + threadIdx.x];

        __syncthreads();

        for (size_t j = 0; j < block_size; ++j) {
            temp = cuda_get_acceleration(temp_bi, arr[j]);

            // acceleration
            accel.x -= temp.x;
            accel.y -= temp.y;
            accel.z -= temp.z;
        }

        __syncthreads();
    }

    // Set positions, velocities
    bodies[i].position.x += (bodies[i].velocity.x + temp_velocity.x) * halftau;
    bodies[i].position.y += (bodies[i].velocity.y + temp_velocity.y) * halftau;
    bodies[i].position.z += (bodies[i].velocity.z + temp_velocity.z) * halftau;

    bodies[i].velocity.x += (accel.x + temp_accel.x) * halftau;
    bodies[i].velocity.y += (accel.y + temp_accel.y) * halftau;
    bodies[i].velocity.z += (accel.z + temp_accel.z) * halftau;
}


void nbody_cuda(
    std::vector<Body> bodies, // we have to copy it since writing would require data to be copied from device back to host
    const T time_interval,
    const size_t iterations,
    const size_t time_layer_export_step,
    const std::string &output_folder,
    const bool benchmark_mode
) {
    const int N = bodies.size();
    const T tau = time_interval / iterations;
    const T halftau = tau / 2;

    // Multiply all masses by G to avoid doing so for every acceleration computation
    ///for (size_t i = 0; i < N; ++i) bodies[i].mass *= G;

    std::vector<std::string> filenames;
    if (!benchmark_mode) {
        // Each body stores it's positions in a separate file 'positions_<index>'
        // using 't  pos.x  pos.y  pos.z' format on each line

        // Setup filenames
        filenames.resize(N);

        const int max_number_width = 1 + static_cast<int>(log10(N));

        for (size_t i = 0; i < N; ++i) {
            const int number_width = 1 + static_cast<int>(log10(i + 1));

            filenames[i] =
                output_folder + "/positions_" +
                std::string(max_number_width - number_width, '0') + std::to_string(i + 1) + ".txt";
        }

        // Reset files and export first time layer
        ensure_folder(output_folder);
        clean_folder(output_folder);
        create_empty_files(filenames);
        export_time_layer(T(0), bodies, filenames);
    }

    // Allocate memory on device and copy bodies to it
    Body* device_bodies_ptr;
    T4* temp_positions_ptr;
    
    cudaError_t err;

    err = cudaMalloc(&device_bodies_ptr, N * sizeof(Body));
    if (err != cudaSuccess) printf("MALLOC ERROR %d %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpy(device_bodies_ptr, bodies.data(), N * sizeof(Body), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("MEMCPY ERROR %d %s\n", err, cudaGetErrorString(err));

    err = cudaMalloc(&temp_positions_ptr, N * sizeof(T4));
    if (err != cudaSuccess) printf("TEMP MALLOC ERROR %d %s\n", err, cudaGetErrorString(err));

    // Setup blocking
    const int blocks = N / CUDA_BLOCK_SIZE;

    // ---------------------------
    // --- Iteration over time ---
    // ---------------------------
    for (int step = 0; step < iterations; step++) {
        // ----------------------
        // --- Explicit Euler ---
        // ----------------------
        cuda_integrate_bodies<<<blocks, CUDA_BLOCK_SIZE>>>(device_bodies_ptr, temp_positions_ptr, N, tau);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) printf("INTEGRATE BODIES ERROR %d %s\n", err, cudaGetErrorString(err));

        // Export time layer with respect to 'TIME_LAYER_EXPORT_STEP'
        if (!benchmark_mode && !((step + 1) % time_layer_export_step)) {
            err = cudaMemcpy(bodies.data(), device_bodies_ptr, N * sizeof(Body), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) printf("FILE WRITE MEMCPY ERROR %d %s\n", err, cudaGetErrorString(err));

            export_time_layer(tau * (step + 1), bodies, filenames);
        }
            
    }
}


// ------------------
// --- CUDA UTILS ---
// ------------------
void print_devices() {
    int numberOfDevices;
    cudaGetDeviceCount(&numberOfDevices);

    for (int i = 0; i < numberOfDevices; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  ------------------------------------\n");
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  ------------------------------------\n");
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  ------------------------------------\n");
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max grid size: {%d, %d, %d}\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  ------------------------------------\n");
        printf("\n");
    }
}

int cuda_finish() {
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

/*
/// BEHOLD NBODY

typedef float T;
typedef float3 T3;
typedef float4 T4;

const T EPSILON = 1e-8;

__device__ T3 add_acceleration(T4 body_i, T4 body_j, T3 a_i) {
    T3 dr;   // r_ij  [3 FLOPS]   
    dr.x = body_j.x - body_i.x;   
    dr.y = body_j.y - body_i.y;   
    dr.z = body_j.z - body_i.z;   

    // dr2 = ||dr||^2 + EPS^2  [6 FLOPS]
    const T dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + EPSILON;

    // dr3inverse = 1/dr2^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    const T dr3inverse = T(1) / (dr2 * sqrtf(dr2));

    // acceleration = m_j * dr3inverse [1 FLOP]
    const T acceleration = body_j.w * dr3inverse;

    // a_i =  a_i + acceleration_ij [6 FLOPS]
    a_i.x += dr.x * acceleration;
    a_i.y += dr.y * acceleration;
    a_i.z += dr.z * acceleration;

    return a_i; 
}

__device__ float3 tile_calculation(float4 myPosition, float3 acceleration) {
    extern __shared__ float4[] sharedPosition;

    for (int i = 0; i < blockDim.x; i++) {
        acceleration = add_acceleration(myPosition, sharedPosition[i], acceleration);
    }

    return acceleration;
}

__global__ void calculate_forces(void *devX, void *devA) {
    extern __shared__ float4[] sharedPosition;

    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;

    
    float3 acc = { 0.0f, 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = globalX[gtid];

    for (int i = 0, tile = 0; i < N; i += p, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;

        sharedPosition[threadIdx.x] = globalX[idx];

        __syncthreads();

        acc = tile_calculation(myPosition, acc);

        __syncthreads();
    }

    // Save the result in global memory for the integration step.
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
} 


/// FROM SAMPLE

void nbody_cuda() {
    int numBlocks = (deviceData[dev].numBodies + blockSize - 1) / blockSize;
    int numTiles = (numBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 4 * sizeof(T);  // 4 floats for pos

    integrateBodies<T> << <numBlocks, blockSize, sharedMemSize >> > (
        (typename vec4<T>::Type *)deviceData[dev].dPos[1 - currentRead],
        (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
        (typename vec4<T>::Type *)deviceData[dev].dVel, deviceData[dev].offset,
        deviceData[dev].numBodies, deltaTime, damping, numTiles);
}*/