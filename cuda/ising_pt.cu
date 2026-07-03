// ising_pt.cu — GPU parallel tempering (replica-exchange Metropolis) for a general Ising model.
//
// The CUDA counterpart of drift/solvers/parallel_tempering.py. That Python file is the oracle:
// this kernel implements the *same* algorithm and must reproduce the exact ground energy on the
// small instances the exact engine can still solve (see cuda/README.md — the falsifier).
//
// Phase 14b — SPARSE J (CSR): a spin's couplings are stored as an adjacency list, so flipping spin
// i updates only its *neighbours'* local fields (O(degree)) instead of all n (the dense O(n) pass
// that Phase 14 measured as the bottleneck). For the sparse graphs these problems actually are,
// this is the single biggest win.
//
// Design — one CUDA block per replica:
//   * A replica holds a spin configuration s (±1) and its local field f_i = (J·s)_i + h_i, so a
//     single-spin flip is O(1) to score (dE = 2·s_i·f_i) and O(degree) to apply (each neighbour j
//     of i shifts by J_ij·Δs_i). The block's threads apply that update in parallel; the neighbours
//     of a spin are distinct, so no atomics are needed.
//   * Replica exchange swaps *temperatures* between adjacent rungs (equivalent to swapping configs,
//     far cheaper); the global best over all replicas is tracked throughout.
//
// Problem/params + CSR come in as one binary file; the best energy + configuration go out as
// another. Build from the x64 Native Tools prompt with cuda/build.bat (nvcc, sm_120).

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

// ── on-disk problem format (little-endian, packed, no struct padding) ─────────
//   uint64 seed
//   int32  n, R, n_rounds, sweeps_per_round, nnz
//   float  T_min, T_max
//   int32  row_ptr[n+1]      (CSR row offsets into col_idx/weight)
//   int32  col_idx[nnz]      (neighbour spin index)
//   float  weight[nnz]       (J_ij coupling of that edge)
//   float  h[n]              (external field)

// ── init: RNG, random spins, local field (via CSR), energy, best ──────────────
__global__ void init_kernel(const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
                            const float* __restrict__ weight, const float* __restrict__ h,
                            float* s, float* field, double* E, double* bestE, float* bestS,
                            curandState* states, int n, uint64_t seed) {
    int r = blockIdx.x;                 // one block per replica
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (tid == 0) curand_init(seed + 1315423911ull * r, 0, 0, &states[r]);
    __syncthreads();

    __shared__ curandState local;
    if (tid == 0) local = states[r];
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < n; ++i) s[r * n + i] = (curand(&local) & 1) ? 1.0f : -1.0f;
        states[r] = local;
    }
    __syncthreads();

    // field_i = sum_{j~i} J_ij s_j + h_i   (CSR row i)
    const float* sr = &s[(size_t)r * n];
    for (int i = tid; i < n; i += stride) {
        float fi = h[i];
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) fi += weight[k] * sr[colIdx[k]];
        field[r * n + i] = fi;
    }
    __syncthreads();

    // E = -0.5 * sum_i s_i * f_i  - 0.5 * sum_i h_i * s_i   (f_i already includes h_i)
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
__global__ void sweep_kernel(const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
                             const float* __restrict__ weight, float* s, float* field,
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
    __shared__ int sh_improve;
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
            // update only the neighbours of i (CSR row i) — the O(degree) win
            for (int k = rowPtr[i] + tid; k < rowPtr[i + 1]; k += stride)
                fr[colIdx[k]] += weight[k] * delta;
        }
        __syncthreads();
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
    int r = 2 * t + parity;
    if (r + 1 >= R) return;
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

template <typename T> static T take(const char*& p) { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <problem.bin> <result.bin>\n", argv[0]);
        return 1;
    }
    std::vector<char> buf = read_file(argv[1]);
    const char* p = buf.data();
    uint64_t seed = take<uint64_t>(p);
    int n = take<int32_t>(p);
    int R = take<int32_t>(p);
    int n_rounds = take<int32_t>(p);
    int sweeps_per_round = take<int32_t>(p);
    int nnz = take<int32_t>(p);
    float T_min = take<float>(p);
    float T_max = take<float>(p);
    const int* rowPtr_h = reinterpret_cast<const int*>(p); p += (size_t)(n + 1) * sizeof(int);
    const int* colIdx_h = reinterpret_cast<const int*>(p); p += (size_t)nnz * sizeof(int);
    const float* weight_h = reinterpret_cast<const float*>(p); p += (size_t)nnz * sizeof(float);
    const float* h_host = reinterpret_cast<const float*>(p);

    // temperature ladder (geometric, ascending) -> beta
    std::vector<float> beta(R);
    for (int r = 0; r < R; ++r) {
        double T = (R == 1) ? T_min
                            : T_min * std::pow((double)T_max / T_min, (double)r / (R - 1));
        beta[r] = (float)(1.0 / (T > 1e-12 ? T : 1e-12));
    }

    int *dRowPtr, *dColIdx;
    float *dWeight, *dh, *ds, *dfield, *dbeta, *dbestS;
    double *dE, *dbestE;
    curandState* dstates;
    CUDA_CHECK(cudaMalloc(&dRowPtr, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dColIdx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dWeight, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dfield, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbestS, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dE, R * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbestE, R * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbeta, R * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dstates, R * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpy(dRowPtr, rowPtr_h, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColIdx, colIdx_h, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWeight, weight_h, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dh, h_host, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta, beta.data(), R * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;   // power of two (the energy reduction needs it); saturates via strided loops

    init_kernel<<<R, threads>>>(dRowPtr, dColIdx, dWeight, dh, ds, dfield, dE, dbestE, dbestS,
                                dstates, n, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n_flips = sweeps_per_round * n;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rnd = 0; rnd < n_rounds; ++rnd) {
        sweep_kernel<<<R, threads>>>(dRowPtr, dColIdx, dWeight, ds, dfield, dE, dbestE, dbestS,
                                     dbeta, dstates, n, n_flips);
        swap_kernel<<<1, (R + 1) / 2>>>(dE, dbeta, dstates, R, rnd & 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double flips = (double)R * n_rounds * n_flips;

    std::vector<double> bestE(R);
    std::vector<float> bestS((size_t)R * n);
    CUDA_CHECK(cudaMemcpy(bestE.data(), dbestE, R * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bestS.data(), dbestS, (size_t)R * n * sizeof(float), cudaMemcpyDeviceToHost));
    int best_r = 0;
    for (int r = 1; r < R; ++r) if (bestE[r] < bestE[best_r]) best_r = r;

    FILE* out = std::fopen(argv[2], "wb");
    std::fwrite(&bestE[best_r], sizeof(double), 1, out);
    std::fwrite(&bestS[(size_t)best_r * n], sizeof(float), n, out);
    std::fclose(out);

    std::fprintf(stderr, "best_E=%.6f  replicas=%d  n=%d  nnz=%d  flips=%.3e  time=%.3fs  throughput=%.3e flips/s\n",
                 bestE[best_r], R, n, nnz, flips, secs, flips / secs);

    cudaFree(dRowPtr); cudaFree(dColIdx); cudaFree(dWeight); cudaFree(dh); cudaFree(ds);
    cudaFree(dfield); cudaFree(dbestS); cudaFree(dE); cudaFree(dbestE); cudaFree(dbeta); cudaFree(dstates);
    return 0;
}
