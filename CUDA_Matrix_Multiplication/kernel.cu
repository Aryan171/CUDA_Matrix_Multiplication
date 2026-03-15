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

	initializeMatrix(a, m * n);
	initializeMatrix(b, n * p);

	// CPU Matrix Multiplication
	int* c = new int[m * p];
	nvtxRangePushA("CPU Matrix Multiplication");
	matrixMultiplyCpu(a, b, c, m, n, p);
	nvtxRangePop();
}