#ifdef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define CUDA_LAUNCH(kernel_name, gridsize, blocksize, ...) \
kernel_name KERNEL_ARGS2(gridsize, blocksize)(__VA_ARGS__);
#endif

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include<cuda_runtime_api.h> 
#include<cuda.h>
#include <device_functions.h>

#define TILE_WIDTH 10

using namespace std;

// Конкретная реализация функции ядра
__global__ void matmul(int* M, int* N, int* Result, int  size)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Col = bx * TILE_WIDTH + tx;
	int Row = by * TILE_WIDTH + ty;

	int Pervalue = 0;

	for (int i = 0; i < (size / TILE_WIDTH); i++) // Сколько TILE_WIDTH там, и каждый цикл вычисляет размер блока
	{
		Mds[ty][tx] = M[Row * size + (i * TILE_WIDTH + tx)];
		Nds[ty][tx] = N[Col + (i * TILE_WIDTH + ty) * size];
		__syncthreads();


		for (int k = 0; k < TILE_WIDTH; k++) // Умножаем TILE_WIDTH
			Pervalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}

	Result[Row * size + Col] = Pervalue;
}

//void print_matrix(int a[][])

int main()
{
	const int Nd = 128;
	int Size = Nd * Nd;
	int* M, * N, * result;
	int width = Nd / 3;

	int a[Nd][Nd];
	int b[Nd][Nd];
	int c[Nd][Nd];

	// Блок потока и разделение потока
	dim3 gridSize(Nd / width, Nd / width);
	dim3 blockSize(width, width);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Распределение памяти устройства
	cudaMalloc((void**)&M, Size * sizeof(int));
	cudaMalloc((void**)&N, Size * sizeof(int));
	cudaMalloc((void**)&result, Size * sizeof(int));

	// Инициализация
	for (int i = 0; i < Nd; i++){
		for (int j = 0; j < Nd; j++){
			a[i][j] = 2;
			b[i][j] = 3;
		}
	}

	// Копирование данных с хоста на устройство
	cudaMemcpy(M, a, Size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(N, b, Size * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	//CUDA_LAUNCH(matmul,gridSize, blockSize, M, N, result, Nd); // Вызов функции ядра
	matmul<<<gridSize, blockSize>>> (M, N, result, Nd);// Вызов функции ядра

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(c, result, Size * sizeof(int), cudaMemcpyDeviceToHost);
	
	cout << "\nTime used:" << elapsedTime;

	cudaFree(M);
	cudaFree(N);
	cudaFree(result);

	return 0;
}
