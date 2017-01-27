#pragma once

#include "cuda_error_check.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
	int current_device = 0, sm_per_multiproc = 0;
	int max_perf_device = 0;
	int device_count = 0, best_SM_arch = 0;
	int devices_prohibited = 0;

	unsigned long long max_compute_perf = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&device_count);

	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&device_count));

	if (device_count == 0)
	{
		fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Find the best major SM Architecture GPU device
	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&deviceProp, current_device);

		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			if (deviceProp.major > 0 && deviceProp.major < 9999)
			{
				best_SM_arch = MAX(best_SM_arch, deviceProp.major);
			}
		}
		else
		{
			devices_prohibited++;
		}

		current_device++;
	}

	if (devices_prohibited == device_count)
	{
		fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
		exit(EXIT_FAILURE);
	}

	// Find the best CUDA capable GPU device
	current_device = 0;

	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&deviceProp, current_device);

		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
			{
				sm_per_multiproc = 1;
			}
			else
			{
				sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
			}

			unsigned long long compute_perf = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

			if (compute_perf > max_compute_perf)
			{
				// If we find GPU with SM major > 2, search only these
				if (best_SM_arch > 2)
				{
					// If our device==dest_SM_arch, choose this, or else pass
					if (deviceProp.major == best_SM_arch)
					{
						max_compute_perf = compute_perf;
						max_perf_device = current_device;
					}
				}
				else
				{
					max_compute_perf = compute_perf;
					max_perf_device = current_device;
				}
			}
		}

		++current_device;
	}

	return max_perf_device;
}

// Check if CUDA requirements are met
int checkCudaRequirements() {
	// Is there a cuda device?
	int device_count = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	if (device_count < 1) {
		fprintf(stderr, "No cuda devices found - we need at least one. \n");
		return 0;
	}
	// We'll be using first device by default
	cudaDeviceProp properties;
	HANDLE_CUDA_ERROR(cudaSetDevice(0));
	HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&properties, 0));
	fprintf(stdout, "Device %d: \"%s\".\n", 0, properties.name);
	fprintf(stdout, "Available global device memory: %llu bytes. \n", properties.totalGlobalMem);
	if (properties.major < 2) {
		fprintf(stderr, "Your cuda device has compute capability %i.%i. We need at least 2.0 for atomic operations. \n", properties.major, properties.minor);
		return 0;
	}
	else {
		fprintf(stdout, "Compute capability: %i.%i.\n", properties.major, properties.minor);
	}
	return 1;
}