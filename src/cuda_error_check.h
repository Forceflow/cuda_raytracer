#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include "cuda_runtime.h"
#include <stdio.h>
#include <cstdlib>

// Function to check CUDA requirements
int checkCudaRequirements();

// Cuda error checking wrapping functions
// Based on CUDA by example book, and http://choorucode.com/2011/03/02/how-to-do-error-checking-in-cuda/

#define CUDA_ERROR_CHECKING_ENABLED // Disable CUDA error checking here if you're feeling DANGEROUS 

#define HANDLE_CUDA_ERROR( err ) __HANDLE_ERROR( err, __FILE__, __LINE__ )
#define CHECK_CUDA_ERROR()    __CHECK_ERROR( __FILE__, __LINE__ )

inline void __HANDLE_ERROR(cudaError err, const char *file, const int line){
#ifdef CUDA_ERROR_CHECKING_ENABLED
	if (cudaSuccess != err){
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

inline void __CHECK_ERROR(const char *file, const int line){
#ifdef CUDA_ERROR_CHECKING_ENABLED
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err){
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance. Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
	    fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
	    exit( -1 );
	}
#endif
	return;
}
#endif