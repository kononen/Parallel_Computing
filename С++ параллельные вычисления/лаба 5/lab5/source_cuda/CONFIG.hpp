#pragma once

// Numeric type
#define DATATYPE_DOUBLE

#ifdef DATATYPE_FLOAT
#define T  float
#define T3 float3
#define T4 float4
#endif

#ifdef DATATYPE_DOUBLE
#define T  double
#define T3 double3
#define T4 double4
#endif



// # Config #
#define TIME_INTERVAL 20
#define ITERATIONS 5
#define TIME_LAYER_EXPORT_STEP 1

#define INPUT_FILENAME     "[input]/{4_body_test}[bodies].txt"
#define OUTPUT_FOLDER      "[output]/[positions]"
#define OUTPUT_FOLDER_CUDA "[output]/{CUDA}[positions]"

#define BENCHMARK_MODE         true
#define TEST_CONVERGENCE_ORDER false
#define USE_RANDOM_BODIES      true

#define SKIP_SERIAL_METHOD true
#define SKIP_CUDA_METHOD   false

#define CUDA_BLOCK_SIZE 256

// # Random config #
#define RANDOM_N     256 * 80 * 10 ///1024 * 18 * 8
#define RANDOM_M_MIN 5e4
#define RANDOM_M_MAX 50e4
#define RANDOM_R_MIN 10
#define RANDOM_R_MAX 100
#define RANDOM_V_MIN 1
#define RANDOM_V_MAX 5
#define RANDOM_FILENAME "[input]/[generated_random_bodies].txt"

// # Convergence test config #
#define OUTPUT_FOLDER_ORDER_TEST_1 "[output]/{cuda_order_test_1}[positions]"
#define OUTPUT_FOLDER_ORDER_TEST_2 "[output]/{cuda_order_test_2}[positions]"
#define OUTPUT_FOLDER_ORDER_TEST_3 "[output]/{cuda_order_test_3}[positions]"
#define q 2

 //2.419e+6; // 28 days


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

__global__ void cuda_integrate_bodies(Body* bodies, T4* temp_positions, int N, T tau) {
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

        for (size_t j = 0; j < block_size; ++j) {
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
    cuda_integrate_bodies << <blocks, CUDA_BLOCK_SIZE >> > (device_bodies_ptr, temp_positions_ptr, N, tau);
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