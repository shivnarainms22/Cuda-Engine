// Reference kernel written by following _WMMA_GUIDANCE (src/cuda_engine/stages/
// codegen.py) LITERALLY. This is not hand-tuned CUDA — deviating from the
// guidance would defeat the purpose. If this fails to compile or produces wrong
// results, the guidance we ship to the LLM is defective.
//
// Covers the fp16 case: row-major inputs, fp32 accumulate, fp16 output,
// ragged (non-multiple-of-16) shapes.

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARPS_PER_BLOCK 4

__global__ void matmul_fp16_wmma(const half* __restrict__ aPtr,
                                 const half* __restrict__ bPtr,
                                 half* __restrict__ cPtr,
                                 int M, int N, int K) {
    const int lda = K, ldb = N, ldc = N;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    const int tilesM = (M + 15) / 16;
    const int tilesN = (N + 15) / 16;

    const int globalWarp = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (globalWarp >= tilesM * tilesN) return;

    const int tileRow = globalWarp / tilesN;
    const int tileCol = globalWarp % tilesN;

    // Guidance: fragments are BOTH row_major (inputs are row-major contiguous).
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
#ifdef NEGATIVE_CONTROL
    // Deliberately reintroduces defect #1 (the col_major matrix_b this guidance
    // was fixed to remove) so we can confirm the harness actually detects it.
    // A green run with this defined would mean the check proves nothing.
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
#else
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
#endif
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Guidance: zero-pad ragged tiles through shared memory. Per-warp buffers.
    __shared__ half aTile[WARPS_PER_BLOCK][16 * 16];
    __shared__ half bTile[WARPS_PER_BLOCK][16 * 16];

    for (int k = 0; k < K; k += 16) {
        for (int i = laneId; i < 16 * 16; i += 32) {
            const int r = i / 16, c = i % 16;
            const int gr = tileRow * 16 + r, gk = k + c;
            aTile[warpId][i] = (gr < M && gk < K) ? aPtr[gr * lda + gk] : __float2half(0.0f);
            const int bk = k + r, gc = tileCol * 16 + c;
            bTile[warpId][i] = (bk < K && gc < N) ? bPtr[bk * ldb + gc] : __float2half(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, &aTile[warpId][0], 16);
        wmma::load_matrix_sync(b_frag, &bTile[warpId][0], 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    // Guidance: fp32 accumulator cannot be stored to half* — stage per-warp and
    // convert with __float2half, guarding the ragged edges.
    __shared__ float stage[WARPS_PER_BLOCK][16 * 16];
    wmma::store_matrix_sync(&stage[warpId][0], acc_frag, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = laneId; i < 16 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        int gr = tileRow * 16 + r, gc = tileCol * 16 + c;
        if (gr < M && gc < N) cPtr[gr * ldc + gc] = __float2half(stage[warpId][i]);
    }
}

extern "C" int launch_matmul_fp16(const void* a, const void* b, void* c, int M, int N, int K) {
    if (M == 0 || N == 0 || K == 0) return 0;  // N = 0 must be a safe no-op
    const int tiles = ((M + 15) / 16) * ((N + 15) / 16);
    const int blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    matmul_fp16_wmma<<<blocks, WARPS_PER_BLOCK * 32>>>(
        (const half*)a, (const half*)b, (half*)c, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    return (int)err;
}
