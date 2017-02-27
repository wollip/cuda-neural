#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

/* Convention
* row = x; col = y
* Matrix are stored row by row
*
*/
namespace wath
{

/**
* Purpose: Generate multiple fibonacci sequence of equal length
*   where the start element is shifted to the right by 1.
* gridDim = number of elements sequences
* blockDim = length of sequence
* Arg outptus = a matrix of size gridDim x blockDim to be filled
*/
__global__ void fibonacci(float** outputs);
 
/**
* Purpose: Generate a sequence of random digits
* gridDim = 1
* blockDim = length of sequence
* Arg outputs = array to be filled
*/
void random(float* outputs);
 
/**
* Purpose: Multiple Matrix by scalar
* gridDim = 1
* blockDim = row, col
* Arg scalar
* Arg input = matrix
* Arg output = matrix
* if output is input then input will be overwritten
*/
__global__ void scalarMatrixMultiply(float* output, float* input, float scalar );
 
/**
* Purpose: Multiple 2 matrix
* gridDim = 1
* blockDim = ( row , col ) of the output matrix
* Arg rightMatrix 
* Arg leftMatrix
* Arg output
* Arg middleLen = the num of cols for the leftMatrix = the num of rows for the rightMatrix
*/
__global__ void matrixMatrixMultiply(float* output, float* leftMatrix, float* rightMatrix, int middleLen);
 
/**
* Purpose: Add 2 matrix
* gridDim = 1
* blockDim = row, col
* Arg leftMatrix
* Arg rightMatrix
* Arg output = same length as left and right matrix, 
* If output reference left or right matrix, that matrix gets replaced.
*/
__global__ void matrixMatrixAdd(float* output, float* leftMatrix, float* rightMatrix);
 
/**
* Purpose: transpose a Matrix
* gridDim = 1
* blockDim = row, col 
* Arg matrix
*/
__global__ void matrixTranspose(float* output, float* input);
 
/**
* Purpose: sum up an array
* gridDim = 1
* blockDim = arrayLength 
* shareSize = arrayLength * sizeof(floats)
* Arg array = 
* Arg sum = 1 index array to store sum
*/
__global__ void sum(float* array, float* d_sum);

// TODO
/* 
sigmoid, tanh, abs and their derivatives
add comments
*/

} // namespace wath
