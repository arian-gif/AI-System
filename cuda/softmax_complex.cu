
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// memory both the CPU and the GPU can read and write, so no cudaMalloc/cudaMemcpy needed
__managed__ float demo_in[8], demo_out[8];

__device__ void dump(const float* s, int n) {
    for (int i = 0; i < n; i++) printf(" %8.4f", s[i]);
    printf("\n");
}

__global__ void softmax_block(const float* x, float* out, int cols) {
    extern __shared__ float scratch[];   // one float per thread, shared by the whole block
    int t = threadIdx.x;

    // ---- step 1: the max of the row ----
    float m = -FLT_MAX;
    for (int j = t; j < cols; j += blockDim.x) m = fmaxf(m, x[j]);
    scratch[t] = m;
    __syncthreads();   // wait until every thread has written its value
    if (t == 0) { printf("  each thread's own max :"); dump(scratch, blockDim.x); }
    __syncthreads();   // (this second wait only keeps the printing tidy)

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) scratch[t] = fmaxf(scratch[t], scratch[t + stride]);
        __syncthreads();
        if (t == 0) { printf("  max, stride %d        :", stride); dump(scratch, blockDim.x); }
        __syncthreads();
    }
    float row_max = scratch[0];   // slot 0 now holds the max of the whole row
    __syncthreads();              // everyone reads slot 0 before we reuse scratch

    // ---- step 2: exp(x - max), and the sum of those ----
    float s = 0.0f;
    for (int j = t; j < cols; j += blockDim.x) {
        float e = expf(x[j] - row_max);
        out[j] = e;   // keep it, step 3 needs it
        s += e;
    }
    scratch[t] = s;
    __syncthreads();
    if (t == 0) { printf("  each thread's own sum :"); dump(scratch, blockDim.x); }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) scratch[t] += scratch[t + stride];
        __syncthreads();
        if (t == 0) { printf("  sum, stride %d        :", stride); dump(scratch, blockDim.x); }
        __syncthreads();
    }
    float row_sum = scratch[0];
    if (t == 0) printf("  max = %.1f, sum of exps = %.4f\n", row_max, row_sum);

    // ---- step 3: divide. each thread only touches elements it wrote in step 2 ----
    for (int j = t; j < cols; j += blockDim.x) out[j] /= row_sum;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    float row[8] = {3, 9, 2, 7, 5, 1, 8, 4};
    for (int j = 0; j < 8; j++) demo_in[j] = row[j];
    printf("row: 3 9 2 7 5 1 8 4   (1 block of 4 threads)\n");
    printf("thread t takes elements t and t+4:  t0 -> 3,5   t1 -> 9,1   t2 -> 2,8   t3 -> 7,4\n\n");

    softmax_block<<<1, 4, 4 * sizeof(float)>>>(demo_in, demo_out, 8);   // 3rd number = shared memory bytes
    CUDA_CHECK(cudaGetLastError());       // was the launch valid?
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for the GPU; this is also when its printf output shows up

    float check = 0.0f;
    printf("\nresult:");
    for (int j = 0; j < 8; j++) { printf(" %.4f", demo_out[j]); check += demo_out[j]; }
    printf("\nprobabilities add up to %.4f (expect 1.0000)\n", check);
    return 0;
}
