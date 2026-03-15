#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <nvtx3/nvToolsExt.h>

#define THREADS_PER_BLOCK 32

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
	m = 1028;
	n = 1028;
	p = 1028;
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
		(p + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
		(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	cudaMatrixMultiplyNaive << <gridSize, dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK) >> > (a, b, c2, m, n, p);

	// Verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	verify(c, c2, m * p);
}