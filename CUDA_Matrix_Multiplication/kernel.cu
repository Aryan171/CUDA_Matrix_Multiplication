#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>

#define BLOCK_DIM 32
#define BLOCK_TILE_SIZE 4

//#define TAKE_USER_INPUT
//#define PERFORM_CPU_MATRIX_MULTIPLICATION

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

__global__ void cudaMatrixMultiply1DBlockTiling(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int m, int n, int p) {
	__shared__ float tileA[BLOCK_DIM][BLOCK_DIM],
		tileB[BLOCK_DIM][BLOCK_DIM];
	float result[BLOCK_TILE_SIZE] = {};

	int rowStart = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_TILE_SIZE,
		column = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < n; i += BLOCK_DIM) {
		for (int j = 0; j < BLOCK_TILE_SIZE; j++) {
			int tileRow = threadIdx.y * BLOCK_TILE_SIZE + j, tileCol = threadIdx.x,
				aRow = rowStart + j, aCol = tileCol + i,
				bRow = tileRow + i, bCol = column;

			tileA[tileRow][tileCol] = (aRow < m && aCol < n) ? a[aRow * n + aCol] : 0;
			tileB[tileRow][tileCol] = (bRow < n && bCol < p) ? b[bRow * p + bCol] : 0;
		}

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j++) {
			for (int k = 0; k < BLOCK_TILE_SIZE; k++) {
				result[k] += tileA[threadIdx.y * BLOCK_TILE_SIZE + k][j] * tileB[j][threadIdx.x];
			}
		}

		__syncthreads();
	}

	if (column < p) {
		for (int i = rowStart, j = 0; i < m && j < BLOCK_TILE_SIZE; i++, j++) {
			c[i * p + column] = result[j];
		}
	}
}

__global__ void cudaMatrixMultiply2DBlockTiling(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int m, int n, int p) {
	__shared__ float tileA[BLOCK_DIM][BLOCK_DIM],
		tileB[BLOCK_DIM][BLOCK_DIM];
	float result[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE] = {};

	int rowStart = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_TILE_SIZE,
		columnStart = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_TILE_SIZE;

	for (int i = 0; i < n; i += BLOCK_DIM) {
		for (int j = 0; j < BLOCK_TILE_SIZE; j++) {
			for (int k = 0; k < BLOCK_TILE_SIZE; k++) {
				int tileRow = threadIdx.y * BLOCK_TILE_SIZE + j,
					tileCol = threadIdx.x * BLOCK_TILE_SIZE + k,
					aRow = rowStart + j, aCol = tileCol + i,
					bRow = tileRow + i, bCol = columnStart + k;
				tileA[tileRow][tileCol] = (aRow < m && aCol < n) ? a[aRow * n + aCol] : 0;
				tileB[tileRow][tileCol] = (bRow < n && bCol < p) ? b[bRow * p + bCol] : 0;
			}
		}

		__syncthreads();

		for (int k = 0; k < BLOCK_TILE_SIZE; k++) {
			for (int l = 0; l < BLOCK_TILE_SIZE; l++) {
				for (int j = 0; j < BLOCK_DIM; j++) {
					result[k][l] += tileA[threadIdx.y * BLOCK_TILE_SIZE + k][j] * tileB[j][threadIdx.x * BLOCK_TILE_SIZE + l];
				}
			}
		}

		__syncthreads();
	}

	for (int i = 0; i < BLOCK_TILE_SIZE && i + rowStart < m; i++) {
		for (int j = 0; j < BLOCK_TILE_SIZE && j + columnStart < p; j++) {
			c[(i + rowStart) * p + j + columnStart] = result[i][j];
		}
	}
}

__global__ void cudaMatrixMultiplyDoubleBuffering(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int m, int n, int p) {
	__shared__ float tileA[2][BLOCK_DIM][BLOCK_DIM],
		tileB[2][BLOCK_DIM][BLOCK_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		column = blockIdx.x * blockDim.x + threadIdx.x;

	float t = 0;

	tileA[0][threadIdx.y][threadIdx.x] = (row < m && threadIdx.x < n) ? a[row * n + threadIdx.x] : 0;
	tileB[0][threadIdx.y][threadIdx.x] = (column < p && threadIdx.y < n) ? b[(threadIdx.y) * p + column] : 0;
	__syncthreads();

	for (int i = 0; i < n; i += BLOCK_DIM) {
		int current = (i / BLOCK_DIM) % 2, next = (current + 1) % 2;

		tileA[next][threadIdx.y][threadIdx.x] = (row < m && threadIdx.x + i + BLOCK_DIM < n) ? a[row * n + threadIdx.x + i + BLOCK_DIM] : 0;
		tileB[next][threadIdx.y][threadIdx.x] = (column < p && threadIdx.y + i + BLOCK_DIM < n) ? b[(threadIdx.y + i + BLOCK_DIM) * p + column] : 0;

		for (int j = 0; j < BLOCK_DIM; j++) {
			t += tileA[current][threadIdx.y][j] * tileB[current][j][threadIdx.x];
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

#ifdef TAKE_USER_INPUT
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

	float* c2;
	cudaMallocManaged(&c2, sizeC);

#ifdef PERFORM_CPU_MATRIX_MULTIPLICATION
	// CPU Matrix Multiplication
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
#endif



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



	// 1D block tiling optimization GPU Matrix Multiplication
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);

	cudaMatrixMultiply1DBlockTiling << <gridSize, dim3(BLOCK_DIM, BLOCK_DIM / BLOCK_TILE_SIZE) >> > (a, b, c2, m, n, p);
	
	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for 1D block tiling implementation\n");
	verify(c, c2, m * p);



	// 2D block tiling optimization GPU Matrix Multiplication
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);

	cudaMatrixMultiply2DBlockTiling << <gridSize, dim3(BLOCK_DIM / BLOCK_TILE_SIZE, BLOCK_DIM / BLOCK_TILE_SIZE) >> > (a, b, c2, m, n, p);

	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for 2D block tiling implementation\n");
	verify(c, c2, m* p);



	// double buffering optimization GPU Matrix Multiplication
	cudaMemPrefetchAsync(c2, sizeC, deviceLocation, 0);

	cudaMatrixMultiplyDoubleBuffering << <gridSize, dim3(BLOCK_DIM, BLOCK_DIM) >> > (a, b, c2, m, n, p);

	// verify the result
	cudaMemPrefetchAsync(c2, sizeC, hostLocation, 0);
	cudaDeviceSynchronize();
	printf("Verifying result for double buffered implementation\n");
	verify(c, c2, m * p);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(c2);
}