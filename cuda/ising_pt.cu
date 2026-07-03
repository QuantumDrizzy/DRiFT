// ising_pt.cu — GPU parallel tempering (replica-exchange Metropolis) for a general Ising model.
//
// The CUDA counterpart of drift/solvers/parallel_tempering.py — the oracle it must agree with:
// it reproduces the exact ground energy on the small instances the exact engine can solve
// (see cuda/README.md). Build with cuda/build.bat (nvcc, sm_120).
//
// Phase 14c — CHECKERBOARD / graph-colouring parallel updates. Single-spin-flip is serial (flipping
// i changes its neighbours' fields), which left the engine latency-bound. But two spins that are
// NOT adjacent can flip at the same time. So the graph is greedily coloured into independent sets
// (no edge within a colour), and a whole colour flips in parallel: each of its spins reads its
// neighbours' *current* spins (all in other colours, hence stable), scores dE = 2 s_i (Σ_{j~i} J_ij
// s_j + h_i), and decides independently. A sweep is then k colour-steps (k = #colours, small for a
// sparse graph) instead of n serial flips — the latency wall of Phase 14b, removed.
//
// No stored field: dE is recomputed from CSR neighbours each time (they're stable within a colour).
// Energy is recomputed exactly (O(nnz)) once per sweep for replica exchange + best-tracking, so
// best_E is always the true energy of best_s (no incremental fp drift) — which the exact-match
// acceptance test needs. One CUDA block per replica; one cuRAND state per thread.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <curand_kernel.h>

#define BDIM 256   // block size (power of two — the energy reduction needs it)

#define CUDA_CHECK(call)                                                              \
    do {                                                                             \
        cudaError_t _e = (call);                                                     \
        if (_e != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                         cudaGetErrorString(_e), __FILE__, __LINE__);                \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// ── on-disk problem format (little-endian, packed) ────────────────────────────
//   uint64 seed
//   int32  n, R, n_rounds, sweeps_per_round, nnz, k
//   float  T_min, T_max
//   int32  row_ptr[n+1] · col_idx[nnz] · float weight[nnz] · float h[n]
//   int32  color_ptr[k+1] · color_spins[n]   (spins grouped by colour)

__device__ __forceinline__ double neigh_sum(int i, const int* rowPtr, const int* colIdx,
                                             const float* weight, const float* sr) {
    double acc = 0.0;
    for (int t = rowPtr[i]; t < rowPtr[i + 1]; ++t) acc += (double)weight[t] * (double)sr[colIdx[t]];
    return acc;
}

// E = -0.5 Σ_i s_i (Σ_{j~i} J_ij s_j) - Σ_i h_i s_i   (block reduction over spins)
__device__ double replica_energy(const int* rowPtr, const int* colIdx, const float* weight,
                                 const float* h, const float* sr, int n, double* red) {
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int i = tid; i < n; i += BDIM) {
        double si = sr[i];
        acc += -0.5 * si * neigh_sum(i, rowPtr, colIdx, weight, sr) - (double)h[i] * si;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = BDIM / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    return red[0];
}

__global__ void init_kernel(const int* rowPtr, const int* colIdx, const float* weight,
                            const float* h, float* s, double* E, double* bestE, float* bestS,
                            curandState* states, int n, uint64_t seed) {
    int r = blockIdx.x, tid = threadIdx.x;
    int gid = r * BDIM + tid;
    curand_init(seed, gid, 0, &states[gid]);
    float* sr = &s[(size_t)r * n];
    for (int i = tid; i < n; i += BDIM) sr[i] = (curand(&states[gid]) & 1) ? 1.0f : -1.0f;
    __syncthreads();

    __shared__ double red[BDIM];
    double e = replica_energy(rowPtr, colIdx, weight, h, sr, n, red);
    if (tid == 0) { E[r] = e; bestE[r] = e; }
    for (int i = tid; i < n; i += BDIM) bestS[(size_t)r * n + i] = sr[i];
}

__global__ void sweep_kernel(const int* rowPtr, const int* colIdx, const float* weight,
                             const float* h, const int* colorPtr, const int* colorSpins, int k,
                             float* s, double* E, double* bestE, float* bestS, const float* beta,
                             curandState* states, int n, int sweeps) {
    int r = blockIdx.x, tid = threadIdx.x;
    int gid = r * BDIM + tid;
    float invT = beta[r];
    curandState st = states[gid];
    float* sr = &s[(size_t)r * n];

    for (int sw = 0; sw < sweeps; ++sw) {
        for (int c = 0; c < k; ++c) {
            // every spin of colour c, in parallel — its neighbours are other colours (stable)
            for (int idx = colorPtr[c] + tid; idx < colorPtr[c + 1]; idx += BDIM) {
                int i = colorSpins[idx];
                double local = (double)h[i] + neigh_sum(i, rowPtr, colIdx, weight, sr);
                double dE = 2.0 * (double)sr[i] * local;
                if (dE <= 0.0 || curand_uniform(&st) < expf((float)(-dE * invT))) sr[i] = -sr[i];
            }
            __syncthreads();   // finish this colour before the next reads it
        }
    }
    states[gid] = st;

    // exact energy for swap + best (no incremental drift)
    __shared__ double red[BDIM];
    double e = replica_energy(rowPtr, colIdx, weight, h, sr, n, red);
    __shared__ int improve;
    if (tid == 0) { E[r] = e; improve = (e < bestE[r]) ? 1 : 0; if (improve) bestE[r] = e; }
    __syncthreads();
    if (improve) for (int i = tid; i < n; i += BDIM) bestS[(size_t)r * n + i] = sr[i];
}

__global__ void swap_kernel(double* E, float* beta, curandState* states, int R, int parity) {
    int t = threadIdx.x, r = 2 * t + parity;
    if (r + 1 >= R) return;
    double d = ((double)beta[r] - (double)beta[r + 1]) * (E[r] - E[r + 1]);
    float u = curand_uniform(&states[r * BDIM]);
    if (d >= 0.0 || u < expf((float)d)) { float tmp = beta[r]; beta[r] = beta[r + 1]; beta[r + 1] = tmp; }
}

static std::vector<char> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (std::fread(buf.data(), 1, sz, f) != (size_t)sz) std::exit(1);
    std::fclose(f);
    return buf;
}
template <typename T> static T take(const char*& p) { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <problem.bin> <result.bin>\n", argv[0]); return 1; }
    std::vector<char> buf = read_file(argv[1]);
    const char* p = buf.data();
    uint64_t seed = take<uint64_t>(p);
    int n = take<int32_t>(p), R = take<int32_t>(p), n_rounds = take<int32_t>(p);
    int sweeps_per_round = take<int32_t>(p), nnz = take<int32_t>(p), k = take<int32_t>(p);
    float T_min = take<float>(p), T_max = take<float>(p);
    const int* rowPtr_h = reinterpret_cast<const int*>(p);   p += (size_t)(n + 1) * sizeof(int);
    const int* colIdx_h = reinterpret_cast<const int*>(p);   p += (size_t)nnz * sizeof(int);
    const float* weight_h = reinterpret_cast<const float*>(p); p += (size_t)nnz * sizeof(float);
    const float* h_host = reinterpret_cast<const float*>(p); p += (size_t)n * sizeof(float);
    const int* colorPtr_h = reinterpret_cast<const int*>(p); p += (size_t)(k + 1) * sizeof(int);
    const int* colorSpins_h = reinterpret_cast<const int*>(p);

    std::vector<float> beta(R);
    for (int r = 0; r < R; ++r) {
        double T = (R == 1) ? T_min : T_min * std::pow((double)T_max / T_min, (double)r / (R - 1));
        beta[r] = (float)(1.0 / (T > 1e-12 ? T : 1e-12));
    }

    int *dRowPtr, *dColIdx, *dColorPtr, *dColorSpins;
    float *dWeight, *dh, *ds, *dbeta, *dbestS;
    double *dE, *dbestE;
    curandState* dstates;
    CUDA_CHECK(cudaMalloc(&dRowPtr, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dColIdx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dWeight, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dColorPtr, (size_t)(k + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dColorSpins, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ds, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbestS, (size_t)R * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dE, R * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbestE, R * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbeta, R * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dstates, (size_t)R * BDIM * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpy(dRowPtr, rowPtr_h, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColIdx, colIdx_h, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWeight, weight_h, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dh, h_host, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColorPtr, colorPtr_h, (size_t)(k + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColorSpins, colorSpins_h, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta, beta.data(), R * sizeof(float), cudaMemcpyHostToDevice));

    init_kernel<<<R, BDIM>>>(dRowPtr, dColIdx, dWeight, dh, ds, dE, dbestE, dbestS, dstates, n, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rnd = 0; rnd < n_rounds; ++rnd) {
        sweep_kernel<<<R, BDIM>>>(dRowPtr, dColIdx, dWeight, dh, dColorPtr, dColorSpins, k,
                                  ds, dE, dbestE, dbestS, dbeta, dstates, n, sweeps_per_round);
        swap_kernel<<<1, (R + 1) / 2>>>(dE, dbeta, dstates, R, rnd & 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double flips = (double)R * n_rounds * sweeps_per_round * n;

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

    std::fprintf(stderr, "best_E=%.6f  replicas=%d  n=%d  nnz=%d  colours=%d  flips=%.3e  time=%.3fs  throughput=%.3e flips/s\n",
                 bestE[best_r], R, n, nnz, k, flips, secs, flips / secs);

    cudaFree(dRowPtr); cudaFree(dColIdx); cudaFree(dWeight); cudaFree(dh); cudaFree(dColorPtr);
    cudaFree(dColorSpins); cudaFree(ds); cudaFree(dbestS); cudaFree(dE); cudaFree(dbestE);
    cudaFree(dbeta); cudaFree(dstates);
    return 0;
}
