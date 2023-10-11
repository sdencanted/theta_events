/* Implementations of CUDA test functions.
 *
 * * * * * * * * * * * *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Stephen Sorley
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * * * * * * * * * * * *
 */
#include "key.h"


#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <vector>

__global__
void get_key_(size_t N, int *x, int *y) {
    size_t thread_grid_idx     = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for(size_t i = thread_grid_idx; i<N; i += num_threads_in_grid) {
        y[i] = x[i] + y[i]*1280;
    }
}

__global__
void rescale_(int *mat, uint8_t *mat_out,float scale) {
    size_t thread_grid_idx     = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for(size_t i = thread_grid_idx; i<1024*768; i += num_threads_in_grid) {
        mat_out[i] = mat[i]*scale;
    }
}
void get_key(size_t N, int *d_x, int *d_y) {
    const int num_sm            = 8; // Jetson Orin NX
    const int blocks_per_sm     = 4;
    const int threads_per_block = 128;
    get_key_<<<blocks_per_sm*num_sm, threads_per_block>>>(N, d_x, d_y);
}



void rescale(int *mat,uint8_t *mat_out, float scale) {
    const int num_sm            = 8; // Jetson Orin NX
    const int blocks_per_sm     = 4;
    const int threads_per_block = 128;
    rescale_<<<blocks_per_sm*num_sm, threads_per_block>>>(mat,mat_out,scale);
}