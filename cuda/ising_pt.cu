// ising_pt.cu — GPU parallel tempering (replica-exchange Metropolis) for a general Ising model.
//
// The CUDA counterpart of drift/solvers/parallel_tempering.py. That Python file is the oracle:
// this kernel implements the *same* algorithm and must reproduce the exact ground energy on the
// small instances the exact engine can still solve (see cuda/README.md — the falsifier).
//
// Design — one CUDA block per replica:
//   * A replica holds a spin configuration s (±1) and its "local field" f_i = (J·s)_i + h_i, so a
//     single-spin flip is O(1) to evaluate (dE = 2·s_i·f_i) and O(n) to apply (every f_j shifts by
//     J_ij·Δs_i). The block's threads apply that O(n) field update in parallel.
//   * J is symmetric, so J[j,i] = J[i*n + j] — reading *row i* makes the field update coalesced.
//   * Replica exchange swaps *temperatures* (not configurations) between adjacent rungs, which is
//     equivalent to swapping configs but far cheaper; the global best over all replicas is tracked
//     throughout, so the answer is never lost to a swap.
//
// Problem/params come in as one binary file; the best energy + configuration go out as another.
// Everything is float32 on the device; the host prints throughput (spin-flips/sec).
//
// NOTE: written against the CPU reference but NOT yet compiled/run here — build and benchmark it
// from the "x64 Native Tools Command Prompt for VS 2022" with cuda/build.bat (nvcc, sm_120).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                              \
    do {                                                                             \
        cudaError_t _e = (call);                                                     \
        if (_e != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                         cudaGetErrorString(_e), __FILE__, __LINE__);                \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// ── on-disk problem format (little-endian) ────────────────────────────────────
//   int32 n, int32 R, int32 n_rounds, int32 sweeps_per_round
//   uint64 seed
//   float32 T_min, float32 T_max
//   float32 J[n*n]  (row-major, symmetric, zero diagonal)
//   float32 h[n]
struct Params {
    int32_t n, R, n_rounds, sweeps_per_round;
    uint64_t seed;
    float T_min, T_max;
};

// ── init: RNG, random spins, local field, energy, best ────────────────────────
__global__ void init_kernel(const float* __restrict__ J, const float* __restrict__ h,
                            float* s, float* field, double* E, double* bestE, float* bestS,
                            curandState* states, int n, uint64_t seed) {
    int r = blockIdx.x;                 // one block per replica
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (tid == 0) curand_init(seed + 1315423911ull * r, 0, 0, &states[r]);
    __syncthreads();

    // random ±1 spins (thread 0 draws a per-spin bit stream for reproducibility within a replica)
    __shared__ curandState local;
    if (tid == 0) local = states[r];
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < n; ++i) s[r * n + i] = (curand(&local) & 1) ? 1.0f : -1.0f;
        states[r] = local;
    }
    __syncthreads();

    // field_i = sum_j J[i,j] s_j + h_i  (row i is contiguous)
    for (int i = tid; i < n; i += stride) {
        float fi = h[i];
        const float* Ji = &J[(size_t)i * n];
        const float* sr = &s[(size_t)r * n];
        for (int j = 0; j < n; ++j) fi += Ji[j] * sr[j];
        field[r * n + i] = fi;
    }
    __syncthreads();

    // E = -0.5 * sum_i s_i * f_i  - 0.5 * sum_i h_i * s_i     (since f_i already includes h_i)
    __shared__ double red[1024];
    double acc = 0.0;
    for (int i = tid; i < n; i += stride) {
        double si = s[r * n + i];
        acc += -0.5 * si * (double)field[r * n + i] - 0.5 * (double)h[i] * si;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = stride / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) { E[r] = red[0]; bestE[r] = red[0]; }
    for (int i = tid; i < n; i += stride) bestS[r * n + i] = s[r * n + i];
}

// ── one round of Metropolis sweeps on a replica at temperature 1/beta[r] ───────
__global__ void sweep_kernel(const float* __restrict__ J, float* s, float* field,
                             double* E, double* bestE, float* bestS, const float* beta,
                             curandState* states, int n, int n_flips) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float invT = beta[r];

    __shared__ curandState local;
    __shared__ int sh_i;
    __shared__ int sh_accept;
    __shared__ float sh_olds;
    __shared__ double sh_E;
    if (tid == 0) { local = states[r]; sh_E = E[r]; }
    __syncthreads();

    float* sr = &s[(size_t)r * n];
    float* fr = &field[(size_t)r * n];

    for (int step = 0; step < n_flips; ++step) {
        if (tid == 0) {
            int i = curand(&local) % n;
            float si = sr[i];
            double dE = 2.0 * (double)si * (double)fr[i];   // f already includes h
            int accept = (dE <= 0.0) || (curand_uniform(&local) < expf((float)(-dE * invT)));
            sh_i = i; sh_olds = si; sh_accept = accept;
            if (accept) { sr[i] = -si; sh_E += dE; }
        }
        __syncthreads();
        if (sh_accept) {
            int i = sh_i;
            float delta = -2.0f * sh_olds;                  // new - old
            const float* Ji = &J[(size_t)i * n];            // row i (contiguous, symmetric)
            for (int j = tid; j < n; j += stride) fr[j] += Ji[j] * delta;
        }
        __syncthreads();
        // track the deepest configuration this replica has seen — decide once (tid 0), then all
        // threads act on the shared flag to avoid a race on bestE[r].
        __shared__ int sh_improve;
        if (tid == 0) {
            sh_improve = (sh_accept && sh_E < bestE[r]) ? 1 : 0;
            if (sh_improve) bestE[r] = sh_E;
        }
        __syncthreads();
        if (sh_improve) {
            for (int j = tid; j < n; j += stride) bestS[r * n + j] = sr[j];
        }
        __syncthreads();
    }
    if (tid == 0) { states[r] = local; E[r] = sh_E; }
}

// ── replica exchange: swap temperatures of adjacent rungs (Metropolis on energies) ──
__global__ void swap_kernel(double* E, float* beta, curandState* states, int R, int parity) {
    int t = threadIdx.x;
    int r = 2 * t + parity;             // adjacent pair (r, r+1)
    if (r + 1 >= R) return;
    // accept with prob min(1, exp((beta_r - beta_{r+1}) * (E_r - E_{r+1})))
    double d = ((double)beta[r] - (double)beta[r + 1]) * (E[r] - E[r + 1]);
    float u = curand_uniform(&states[r]);
    if (d >= 0.0 || u < expf((float)d)) {
        float tmp = beta[r]; beta[r] = beta[r + 1]; beta[r + 1] = tmp;
    }
}

static std::vector<char> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (std::fread(buf.data(), 1, sz, f) != (size_t)sz) { std::exit(1); }
    std::fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <problem.bin> <result.bin>\n", argv[0]);
        return 1;
    }
    std::vector<char> buf = read_file(argv[1]);
    const char* p = buf.data();
    Params par; std::memcpy(&par, p, sizeof(Params)); p += sizeof(Params);
    int n = par.n, R = par.R;
    const float* J_host = reinterpret_cast<const float*>(p); p += (size_t)n * n * sizeof(float);
    const float* h_host = reinterpret_cast<const float*>(p);

    // temperature ladder (geometric, ascending) -> beta
    std::vector<float> beta(R);
    for (int r = 0; r < R; ++r) {
        double T = (R == 1) ? par.T_min
                            : par.T_min * std::pow((double)par.T_max / par.T_min, (double)r / (R - 1));
        beta[r] = (float)(1.0 / (T > 1e-12 ? T : 1e-12));
    }

    // device allocations
    float *dJ, *dh, *ds, *dfield, *dbeta, *dbestS;
    double *dE, *dbestE;
    curandState* dstates;
    CUDA_CHECK(cudaMalloc(&dJ, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dfield, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbestS, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dE, R * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbestE, R * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbeta, R * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dstates, R * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpy(dJ, J_host, (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dh, h_host, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta, beta.data(), R * sizeof(float), cudaMemcpyHostToDevice));

    // Power-of-two block size (the energy reduction in init_kernel needs it); 256 saturates the
    // per-replica work for any n via the strided loops, and idle threads for tiny n are harmless.
    int threads = 256;

    init_kernel<<<R, threads>>>(dJ, dh, ds, dfield, dE, dbestE, dbestS, dstates, n, par.seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n_flips = par.sweeps_per_round * n;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rnd = 0; rnd < par.n_rounds; ++rnd) {
        sweep_kernel<<<R, threads>>>(dJ, ds, dfield, dE, dbestE, dbestS, dbeta, dstates, n, n_flips);
        swap_kernel<<<1, (R + 1) / 2>>>(dE, dbeta, dstates, R, rnd & 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double flips = (double)R * par.n_rounds * n_flips;

    // reduce the best over all replicas on the host
    std::vector<double> bestE(R);
    std::vector<float> bestS((size_t)R * n);
    CUDA_CHECK(cudaMemcpy(bestE.data(), dbestE, R * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bestS.data(), dbestS, (size_t)R * n * sizeof(float), cudaMemcpyDeviceToHost));
    int best_r = 0;
    for (int r = 1; r < R; ++r) if (bestE[r] < bestE[best_r]) best_r = r;

    // result file: float64 best_E, then n float32 spins (±1)
    FILE* out = std::fopen(argv[2], "wb");
    std::fwrite(&bestE[best_r], sizeof(double), 1, out);
    std::fwrite(&bestS[(size_t)best_r * n], sizeof(float), n, out);
    std::fclose(out);

    std::fprintf(stderr, "best_E=%.6f  replicas=%d  n=%d  flips=%.3e  time=%.3fs  throughput=%.3e flips/s\n",
                 bestE[best_r], R, n, flips, secs, flips / secs);

    cudaFree(dJ); cudaFree(dh); cudaFree(ds); cudaFree(dfield); cudaFree(dbestS);
    cudaFree(dE); cudaFree(dbestE); cudaFree(dbeta); cudaFree(dstates);
    return 0;
}
