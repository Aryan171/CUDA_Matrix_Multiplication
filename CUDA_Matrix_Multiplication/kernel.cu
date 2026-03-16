#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>

#define BLOCK_DIM 32

#define USER_INPUT

void matrixMultiplyCpu(float* a, float* b, float* c, int m, int n, int p) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			c[i * p + j] = 0;
			for (int k = 0; k < n; k++) {
				c[i * p + j] += a[i * n + k] * b[k * p + j];
			}
		}
	}
}

void initializeMatrix(float* matrix, int size) {
	for (int i = 0; i < size; i++) {
		matrix[i] = rand() % 10;
	}
}

void verify(float* c, float* d, int size) {
	float delta = 1e-3;
	for (int i = 0; i < size; i++) {
		if (abs(c[i] - d[i]) > delta) {
			printf("Verification failed at index %d: %f != %f\n", i, c[i], d[i]);
			return;
		}
	}
	printf("Verification successful!\n");
}

__global__ void cudaMatrixMultiplyNaive(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int m, int n, int p) {
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		column = blockIdx.x * blockDim.x + threadIdx.x,
		index = row * p + column;
	if (row < m && column < p) {
		c[index] = 0.0f;
		for (int i = 0; i < n; i++) {
			c[index] += a[row * n + i] * b[i * p + column];
		}
	}
}

__global__ void cudaMatrixMultiplyTiled(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int m, int n, int p) {
	__shared__ float tileA[BLOCK_DIM][BLOCK_DIM],
		tileB[BLOCK_DIM][BLOCK_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		column = blockIdx.x * blockDim.x + threadIdx.x;

	float t = 0;

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

void matrixMultiplyCublas(float* a, float* b, float* c, int m, int n, int p) {
	cublasHandle_t handle;
	cublasCreate(&handle);

	// C = alpha * (A * B) + beta * C
	const float alpha = 1.0f;
	const float beta = 0.0f;

	nvtxRangePushA("cuBLAS Matrix Multiplication\n");
	cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		p, m, n,           // Dimensions: Note the swap of p and m
		&alpha,
		b, p,            // Matrix B acts as the 'left' matrix
		a, n,            // Matrix A acts as the 'right' matrix
		&beta,
		c, p);           // Leading dimension of output C
	nvtxRangePop();

	cublasDestroy(handle);
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
	m = 5000;
	n = 5000;
	p = 5000;
#endif

	int sizeA = m * n * sizeof(float),
		sizeB = n * p * sizeof(float),
		sizeC = m * p * sizeof(float);

	float* a, * b;

	cudaMallocManaged(&a, m * n * sizeof(float));
	cudaMallocManaged(&b, n * p * sizeof(float));

	cudaMemLocation deviceLocation, hostLocation;
	deviceLocation.type = cudaMemLocationTypeDevice;
	deviceLocation.id = deviceId;
	hostLocation.type = cudaMemLocationTypeHost;
	hostLocation.id = 0;

	initializeMatrix(a, m * n);
	initializeMatrix(b, n * p);

	// cuBLAS GPU Matrix Multiplication
	float* c;
	cudaMallocManaged(&c, sizeC);
	cudaMemPrefetchAsync(c, sizeC, deviceLocation, 0);
	cudaDeviceSynchronize();
	cudaMemPrefetchAsync(a, sizeA, deviceLocation, 0);
	cudaMemPrefetchAsync(b, sizeB, deviceLocation, 0);
	matrixMultiplyCublas(a, b, c, m, n, p);


	// CPU Matrix Multiplication
	float* c2;
	cudaMallocManaged(&c2, sizeC);
	nvtxRangePushA("CPU Matrix Multiplication\n");
	matrixMultiplyCpu(a, b, c2, m, n, p);
	nvtxRangePop();

	cudaMemPrefetchAsync(a, sizeA, deviceLocation, 0);
	cudaMemPrefetchAsync(b, sizeB, deviceLocation, 0);
	cudaDeviceSynchronize();

	// verify the result
	printf("Verifying result for CPU implementation\n");
	cudaDeviceSynchronize();
	verify(c, c2, m * p);



	// naive GPU Matrix Multiplication
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);
	uint3 gridSize = dim3(
		(p + BLOCK_DIM - 1) / BLOCK_DIM,
		(m + BLOCK_DIM - 1) / BLOCK_DIM);
	cudaMatrixMultiplyNaive << <gridSize, dim3(BLOCK_DIM, BLOCK_DIM) >> > (a, b, c2, m, n, p);

	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for naive implementation\n");
	verify(c, c2, m * p);

	// memory tiling optimized GPU Matrix Multiplication
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);

	cudaMatrixMultiplyTiled << <gridSize, dim3(BLOCK_DIM, BLOCK_DIM) >> > (a, b, c2, m, n, p);

	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for tiled implementation\n");
	verify(c, c2, m * p);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(c2);
}