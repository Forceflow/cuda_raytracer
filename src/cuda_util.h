#pragma once
#include "libs/helper_cuda.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Check if CUDA requirements are met
int setBestCUDADevice(int compute_min = 2) {
	// Is there a cuda device?
	int device_count = 0;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	if (device_count < 1) {
		fprintf(stderr, "CUDA: No CUDA devices found. We need at least one. \n");
		exit(0);
	}
	// Get best device and set it
	int best = gpuGetMaxGflopsDeviceId();
	fprintf(stdout, "CUDA: Found %i CUDA devices, and the best one is %i \n", device_count, best);
	cudaDeviceProp properties;
	checkCudaErrors(cudaSetDevice(best));
	checkCudaErrors(cudaGetDeviceProperties(&properties, best));
	fprintf(stdout, "CUDA: Device %d: \"%s\".\n", 0, properties.name);
	fprintf(stdout, "CUDA: Available global device memory: %llu bytes \n", properties.totalGlobalMem);
	fprintf(stdout, "CUDA: Compute capability: %i.%i.\n", properties.major, properties.minor);
	return 1;
}