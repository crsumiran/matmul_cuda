#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define BM 128
#define BK 128
#define BN 8
#define TM 8 // BM divided by BN
#define TK 8 // BK divided by BN
#define MAX(a,b) (a > b) ? a : b
// ─────────────────────────────────────────────
//  Problem dimensions
// ─────────────────────────────────────────────
#define M_DIM  8192
#define N_DIM  6144
#define K_DIM  4096

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define BLOCKSIZE 32

// ─────────────────────────────────────────────
//  Your kernel (unchanged)
// ─────────────────────────────────────────────
/*__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K)
{
    const uint tidx = BLOCKSIZE * blockIdx.x + threadIdx.x % BLOCKSIZE;
    const uint tidy = BLOCKSIZE * blockIdx.y + threadIdx.x / BLOCKSIZE;

    if (tidx < K && tidy < M) {
        float tmp = 0.0f;
        for (int i = 0; i < N; i++)
            tmp += A[tidy * N + i] * B[i * K + tidx];
        C[tidy * K + tidx] = tmp;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 threadsPerBlock(32 * 32);
    dim3 blocksPerGrid(CEIL_DIV(K, 32), CEIL_DIV(M, 32));
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
*/
// In your harness / host code

#define ALIGN_TO 128   // must be >= max(BM, BK, BN)

inline int pad(int x) {
    return ((x + ALIGN_TO - 1) / ALIGN_TO) * ALIGN_TO;
}

// Allocate a padded device matrix and copy real data into it
// Real data is (rows × cols), stored row-major with stride real_cols
// Padded allocation is (rows_pad × cols_pad), zero-filled
float* alloc_padded(const float* h_src,
                    int real_rows, int real_cols,
                    int pad_rows,  int pad_cols)
{
    float* d_ptr = nullptr;
    size_t bytes = (size_t)pad_rows * pad_cols * sizeof(float);

    cudaMalloc(&d_ptr, bytes);
    cudaMemset(d_ptr, 0, bytes);   // zero-fill so padding contributes 0 to dot products

    // Copy each real row into the padded allocation
    // (rows have different strides: real_cols vs pad_cols)
    cudaMemcpy2D(
        d_ptr,                          // dst: padded device buffer
        pad_cols * sizeof(float),       // dst stride (bytes per row)
        h_src,                          // src: host buffer
        real_cols * sizeof(float),      // src stride (bytes per row)
        real_cols * sizeof(float),      // width to copy (bytes)
        real_rows,                      // number of rows
        cudaMemcpyHostToDevice
    );

    return d_ptr;
}

// Implement with all conditions for my learning and the next version
// will be a version where matrix sizes are aligned to BM BK to avoid checks
// We will be invoking 512 threads BM * BK / TM number of threads per block
// each thread will calculate TM number of outputs
__global__ void matrix_multiplication_kernel_dwarp_tiling(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
   //const uint tidx = blockDim.x*blockIdx.x + threadIdx.x;
   //const uint tidy = blockDim.y*blockIdx.y + threadIdx.y;
    // Which tile of C this block owns
    int cTileRow = blockIdx.y;
    int cTileCol = blockIdx.x;

    int threadRow = threadIdx.x / (BK/TK); // 256 threads split as 16x16
    int threadCol = threadIdx.x % (BK/TK);

    // Shared memory tiles
    __shared__ float As[BM * BN];
    __shared__ float Bs[BN * BK];
    int numthreads = BM*BK/(TM*TK);
    int aStride = numthreads/BN;
    int bStride = numthreads/BK;

    // Advance base pointers to this block's starting position
    A += cTileRow * BM * N;      // A: move to row-strip for cRow
    B += cTileCol * BK;          // B: move to col-strip for cCol
    C += cTileRow * BM * K + cTileCol*BK;

    int aLocalCol = threadIdx.x%BN;
    int aLocalRow = threadIdx.x/BN;

    int bLocalCol = threadIdx.x%BK;
    int bLocalRow = threadIdx.x/BK;


    float threadRes[TM * TK] = {0.0f};
    float regA[TM] = {0.0f};
    float regB[TK] = {0.0f};

    for (int blkIdx =0; blkIdx < N; blkIdx += BN)
    {
        for (int asidx =0; asidx < BM ; asidx += aStride)
        {
            As[(aLocalRow +asidx)*BN + aLocalCol] =  A[(aLocalRow + asidx)*N + aLocalCol];
        }
        for (int bsidx =0; bsidx < BN ; bsidx += bStride)
        {
            Bs[(bLocalRow + bsidx)*BK + bLocalCol] =  B[(bLocalRow + bsidx)*K + bLocalCol];
        }
        __syncthreads();

        A += BN;
        B += BN*K;

        for (int dotIdx =0 ; dotIdx < BN; dotIdx++)
        {
            for (int regAidx =0; regAidx < TM; regAidx ++){
                regA[regAidx] = As[(threadRow*TM + regAidx)*BN + dotIdx];
            }

            for (int regBidx =0; regBidx < TK; regBidx ++){
                regB[regBidx] = Bs[threadCol*TK + dotIdx*BK + regBidx];
            }

            for (int resAIdx = 0; resAIdx < TM; resAIdx++){
                for (int resBIdx = 0; resBIdx < TK; resBIdx++){
                    threadRes[resAIdx*TK + resBIdx] += regA[resAIdx]*regB[resBIdx];
                }
            }

        }
        __syncthreads();

    }
    for (int resAIdx = 0; resAIdx < TM; resAIdx++){
        for (int resBIdx = 0; resBIdx < TK; resBIdx++){
            C[(threadRow * TM + resAIdx) * K + threadCol * TK + resBIdx] = threadRes[resAIdx*TK + resBIdx];
        }
    }
    //C[(cGlobalRow)*K + cGlobalCol] = threadRes[0];
    //C[(cGlobalRow+1)*K + cGlobalCol] = threadRes[1];
    //C[(cGlobalRow+2)*K + cGlobalCol] = threadRes[2];
    //C[(cGlobalRow+3)*K + cGlobalCol] = threadRes[3];
    //C[(cGlobalRow+4)*K + cGlobalCol] = threadRes[4];
    //C[(cGlobalRow+5)*K + cGlobalCol] = threadRes[5];
    //C[(cGlobalRow+6)*K + cGlobalCol] = threadRes[6];
    //C[(cGlobalRow+7)*K + cGlobalCol] = threadRes[7];


}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
//extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
//    dim3 threadsPerBlock(BM*BK/TM);
//    // number of blocks will be same assuming BK BM but the number of threads per block will be less because each
//    // thread will compute TM number of outputs
//    dim3 blocksPerGrid((K + BK - 1) / BK,
//                       (M + BM - 1) / BM);
//
//
//    matrix_multiplication_kernel_dwarp_tiling<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
//    cudaDeviceSynchronize();
//}

extern "C" void solve(const float* hA, const float* hB, float* hC,
                  int M, int N, int K)
{
    if ((M%ALIGN_TO ==0) && ( N% ALIGN_TO == 0) && (K% ALIGN_TO==0)){
            // Launch with padded dimensions — kernel never sees M/N/K again
        dim3 threads(BM * BK / TM/TK);
        dim3 blocks(K / BK, M / BM);   // exact division guaranteed

            //matrix_multiplication_kernel_no_check
        matrix_multiplication_kernel_dwarp_tiling<<<blocks, threads>>>(hA, hB, hC, M, N, K);
        cudaDeviceSynchronize();

    } else {
        int Mp = pad(M), Np = pad(N), Kp = pad(K);

        float *dA, *dB, *dC;
        dA = alloc_padded(hA, M, N, Mp, Np);
        dB = alloc_padded(hB, N, K, Np, Kp);

            // C is output only — just allocate zeroed
        cudaMalloc(&dC, (size_t)Mp * Kp * sizeof(float));
        cudaMemset(dC, 0, (size_t)Mp * Kp * sizeof(float));

            // Launch with padded dimensions — kernel never sees M/N/K again
        dim3 threads(BM * BK / TM/TK);
        dim3 blocks(Kp / BK, Mp / BM);   // exact division guaranteed

            //matrix_multiplication_kernel_no_check
        matrix_multiplication_kernel_dwarp_tiling<<<blocks, threads>>>(dA, dB, dC, Mp, Np, Kp);
        cudaDeviceSynchronize();

            // Copy real result rows back out (padded stride → real stride)
        cudaMemcpy2D(
            hC,                        // dst host buffer
            K * sizeof(float),         // dst stride
            dC,                        // src padded device buffer
            Kp * sizeof(float),        // src stride
            K * sizeof(float),         // width to copy
            M,                         // number of rows
            cudaMemcpyDeviceToHost
        );

            cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}
// ─────────────────────────────────────────────
//  cuBLAS reference  C = A * B
//  A: M×N   B: N×K   C: M×K  (row-major)
//  cuBLAS is column-major, so we compute
//    C^T = B^T * A^T  with sgemm
// ─────────────────────────────────────────────
void cublas_matmul(cublasHandle_t handle,
                   const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    const float alpha = 1.0f, beta = 0.0f;
    // sgemm(handle, transB, transA, K, M, N, &alpha, B, K, A, N, &beta, C, K)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                K, M, N,
                &alpha,
                B, K,   // B^T treated as column-major K×N
                A, N,   // A^T treated as column-major N×M
                &beta,
                C, K);
}

// ─────────────────────────────────────────────
//  Correctness check
// ─────────────────────────────────────────────
bool check_correctness(const float* ref, const float* out,
                       long long total, float rel_tol = 1e-3f)
{
    long long errors = 0;
    float max_rel_err = 0.0f;

    for (long long i = 0; i < total; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float base = fmaxf(fabsf(ref[i]), 1e-6f);
        float rel  = diff / base;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > rel_tol) {
            if (errors < 5)   // print first few mismatches
                printf("  MISMATCH at [%lld]: ref=%.6f  got=%.6f  rel_err=%.4e\n",
                       i, ref[i], out[i], rel);
            errors++;
        }
    }

    printf("Max relative error : %.4e\n", max_rel_err);
    printf("Mismatches (>%.0e): %lld / %lld\n", (double)rel_tol, errors, total);
    return errors == 0;
}

// ─────────────────────────────────────────────
//  Timing helper (GPU events)
// ─────────────────────────────────────────────
static inline float time_kernel(void (*fn)(const float*, const float*, float*, int, int, int),
                                 const float* dA, const float* dB, float* dC,
                                 int M, int N, int K,
                                 int warmup, int iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; i++) fn(dA, dB, dC, M, N, K);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) fn(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main()
{
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    printf("=== MatMul Harness  M=%d  N=%d  K=%d ===\n\n", M, N, K);

    // ── Host allocations ──────────────────────
    long long szA = (long long)M * N;
    long long szB = (long long)N * K;
    long long szC = (long long)M * K;

    float* hA  = (float*)malloc(szA * sizeof(float));
    float* hB  = (float*)malloc(szB * sizeof(float));
    float* hC  = (float*)malloc(szC * sizeof(float));
    float* hRef = (float*)malloc(szC * sizeof(float));

    // ── Initialise with a known pattern ───────
    // A[i][j] = sin(i + j*0.1)   (values in [-1,1], full rank)
    // B[i][j] = cos(i*0.1 + j)
    srand(42);
    for (long long i = 0; i < szA; i++)
        hA[i] = sinf((float)(i % M) + (float)(i / M) * 0.1f);
    for (long long i = 0; i < szB; i++)
        hB[i] = cosf((float)(i % N) * 0.1f + (float)(i / N));

    // ── Device allocations ────────────────────
    float *dA, *dB, *dC, *dRef;
    cudaMalloc(&dA,  szA * sizeof(float));
    cudaMalloc(&dB,  szB * sizeof(float));
    cudaMalloc(&dC,  szC * sizeof(float));
    cudaMalloc(&dRef, szC * sizeof(float));

    cudaMemcpy(dA, hA, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB * sizeof(float), cudaMemcpyHostToDevice);

    // ── cuBLAS reference ─────────────────────
    cublasHandle_t handle;
    cublasCreate(&handle);
    printf("[1/4] Running cuBLAS reference ... ");
    fflush(stdout);
    cublas_matmul(handle, dA, dB, dRef, M, N, K);
    cudaDeviceSynchronize();
    printf("done.\n");

    // ── Your kernel ───────────────────────────
    printf("[2/4] Running your kernel        ... ");
    fflush(stdout);
    solve(dA, dB, dC, M, N, K);
    printf("done.\n");

    // ── Copy back & check ─────────────────────
    cudaMemcpy(hRef, dRef, szC * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC,   dC,   szC * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n[3/4] Correctness check:\n");
    bool ok = check_correctness(hRef, hC, szC);
    printf("Result: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    // ── Benchmark ─────────────────────────────
    printf("[4/4] Benchmarking (5 warmup, 20 timed iterations):\n");

    // Wrap cublas in a compatible signature for the timer
    // We'll time manually for cuBLAS
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int warmup = 5, iters = 20;
    for (int i = 0; i < warmup; i++) cublas_matmul(handle, dA, dB, dRef, M, N, K);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) cublas_matmul(handle, dA, dB, dRef, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_cublas;
    cudaEventElapsedTime(&ms_cublas, t0, t1);
    ms_cublas /= iters;

    float ms_yours = time_kernel(solve, dA, dB, dC, M, N, K, warmup, iters);

    // TFLOP/s = 2*M*N*K / (time_s * 1e12)
    double ops = 2.0 * M * N * K;
    double tflops_cublas = ops / (ms_cublas * 1e-3) / 1e12;
    double tflops_yours  = ops / (ms_yours  * 1e-3) / 1e12;

    printf("  cuBLAS   : %7.3f ms  →  %.3f TFLOP/s\n", ms_cublas, tflops_cublas);
    printf("  Your kernel: %7.3f ms  →  %.3f TFLOP/s\n", ms_yours,  tflops_yours);
    printf("  Efficiency vs cuBLAS: %.1f%%\n\n",
           100.0 * tflops_yours / tflops_cublas);

    // ── Cleanup ───────────────────────────────
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    free(hA); free(hB); free(hC); free(hRef);

    return ok ? 0 : 1;
}

