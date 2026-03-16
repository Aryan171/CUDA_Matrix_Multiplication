#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <nvtx3/nvToolsExt.h>

#define BLOCK_DIM 32

//#define USER_INPUT

void matrixMultiplyCpu(int* a, int* b, int* c, int m, int n, int p) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			c[i * p + j] = 0;
			for (int k = 0; k < n; k++) {
				c[i * p + j] += a[i * n + k] * b[k * p + j];
			}
		}
	}
}

void initializeMatrix(int* matrix, int size) {
	for (int i = 0; i < size; i++) {
		matrix[i] = rand() % 10;
	}
}

void verify(int* c, int* d, int size) {
	for (int i = 0; i < size; i++) {
		if (c[i] != d[i]) {
			printf("Verification failed at index %d: %d != %d\n", i, c[i], d[i]);
			return;
		}
	}
	printf("Verification successful!\n");
}

__global__ void cudaMatrixMultiplyNaive(int* a, int* b, int* c, int m, int n, int p) {
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		column = blockIdx.x * blockDim.x + threadIdx.x,
		index = row * p + column;
	if (row < m && column < p) {
		c[index] = 0;
		for (int i = 0; i < n; i++) {
			c[index] += a[row * n + i] * b[i * p + column];
		}
	}
}

__global__ void cudaMatrixMultiplyTiled(int* a, int* b, int* c, int m, int n, int p) {
	__shared__ int tileA[BLOCK_DIM][BLOCK_DIM],
		tileB[BLOCK_DIM][BLOCK_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		column = blockIdx.x * blockDim.x + threadIdx.x;

	int t = 0;

	for (int i = 0; i < n; i += BLOCK_DIM) {
		tileA[threadIdx.y][threadIdx.x] = (row < m && threadIdx.x + i < n) ? a[row * n + threadIdx.x + i] : 0;
		tileB[threadIdx.y][threadIdx.x] = (column < p && threadIdx.y + i < n) ? b[(threadIdx.y + i) * p + column] : 0;

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j++) {
			t += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < m && column < p) {
		c[row * p + column] = t;
	}
}

int main()
{
	int m, n, p, deviceId;
	cudaGetDevice(&deviceId);

#ifdef USER_INPUT
	printf("Enter the number of rows and columns of the matrix a ");
	scanf("%d", &m);
	scanf("%d", &n);

	printf("Enter the number of columns of matrix b ");
	scanf("%d", &p);
#else
	m = 1024;
	n = 1024;
	p = 1024;
#endif

	int sizeA = m * n * sizeof(int),
		sizeB = n * p * sizeof(int),
		sizeC = m * p * sizeof(int);

	int* a, * b;

	cudaMallocManaged(&a, m * n * sizeof(int));
	cudaMallocManaged(&b, n * p * sizeof(int));

	cudaMemLocation deviceLocation, hostLocation;
	deviceLocation.type = cudaMemLocationTypeDevice;
	deviceLocation.id = deviceId;
	hostLocation.type = cudaMemLocationTypeHost;
	hostLocation.id = 0;

	initializeMatrix(a, m * n);
	initializeMatrix(b, n * p);

	// CPU Matrix Multiplication
	int* c = new int[m * p];
	nvtxRangePushA("CPU Matrix Multiplication");
	matrixMultiplyCpu(a, b, c, m, n, p);
	nvtxRangePop();

	cudaMemPrefetchAsync(a, sizeA, deviceLocation, 0);
	cudaMemPrefetchAsync(b, sizeB, deviceLocation, 0);
	cudaDeviceSynchronize();

	// naive GPU Matrix Multiplication
	int* c2;
	cudaMallocManaged(&c2, sizeC);
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);

	uint3 gridSize = dim3(
		(p + BLOCK_DIM - 1) / BLOCK_DIM,
		(m + BLOCK_DIM - 1) / BLOCK_DIM);
	cudaMatrixMultiplyNaive << <gridSize, dim3(BLOCK_DIM, BLOCK_DIM) >> > (a, b, c2, m, n, p);

	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for naive implementation");
	verify(c, c2, m * p);

	// optimized GPU Matrix Multiplication
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);

	cudaMatrixMultiplyTiled << <gridSize, dim3(BLOCK_DIM, BLOCK_DIM) >> > (a, b, c2, m, n, p);

	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for tiled implementation");
	verify(c, c2, m * p);
}