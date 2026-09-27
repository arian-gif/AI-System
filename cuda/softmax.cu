#include <stdio.h>
#include <math.h>

#define ROWS 2
#define COLS 3

__managed__ float score[ROWS*COLS];
__managed__ float probs[ROWS*COLS];


__global__ void softmax(float* x, float* out, int cols) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;   
    float* in = x + row * cols;                        
    float* o = out + row * cols;                      


    float m = in[0];
    for (int j = 1; j < cols; j++) {
        if (in[j] > m) m = in[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        o[j] = expf(in[j] - m);
        sum += o[j];
    }

    for (int j = 0; j < cols; j++) {
        o[j] = o[j] / sum;
    }
}

int main() {
    for (int j = 0; j < COLS; j++) {
        scores[j] = j + 1;             
        scores[COLS + j] = 1000 + j;  
    }

    softmax<<<1, ROWS>>>(scores, probs, COLS);  
    cudaDeviceSynchronize();

    for (int r = 0; r < ROWS; r++) {
        printf("row %d:", r);
        for (int j = 0; j < COLS; j++) printf(" %.4f", probs[r * COLS + j]);
        printf("\n");
    }
    return 0;
}
