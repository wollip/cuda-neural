#include "cudaLayer/Math.cuh"

namespace wath
{

__global__ void fibonacci(float* outputs)
{
	float n = float(blockIdx.x + threadIdx.x);
	outputs[blockIdx.x * blockDim.x + threadIdx.x] = ((5 + 3 * pow(5.0, 0.5)) / 10) * pow(float((1 + pow(5.0, 0.5)) / 2), n) + ((5 - 3 * pow(5.0, 0.5)) / 10) * pow(float((1 - pow(5.0, 0.5)) / 2), n);
}

__global__ void scalarMatrixMultiply(float* output, float* input, float scalar)
{
	int index = threadIdx.y + threadIdx.x * blockDim.y;
	output[index] = scalar * input[index];
}

__global__ void matrixMatrixMultiply(float* output, float* leftMatrix, float* rightMatrix, int middleLen)
{
	float store = 0;
	int row = threadIdx.x;
	int col = threadIdx.y;

	for( int i = 0; i < middleLen; ++i)
	{
		store += leftMatrix[row * middleLen + i] * rightMatrix[col + i * blockDim.y];
	}

	output[ threadIdx.x * blockDim.y + threadIdx.y] = store;
}

__global__ void matrixMatrixAdd(float* output, float* leftMatrix, float* rightMatrix)
{
	int index = threadIdx.y + threadIdx.x * blockDim.y;
	output[index] = leftMatrix[index] + rightMatrix[index];
}

__global__ void sum(float* array, float* output)
{
	extern __shared__ float s[];
	int index = threadIdx.x;

	s[index] = array[index];
	__syncthreads();

	for (int i = 1; i < blockDim.x; i <<= 1){
		if (index%(i*2) == 0){
			s[index] += s[index + i];
		}
		__syncthreads();
	}
	if (index == 0){
		output[0] = s[0];
	}
}

__global__ void matrixTranspose(float* output, float* input)
{
	int store = input[threadIdx.y + threadIdx.x * blockDim.y];
	__syncthreads();
	output[threadIdx.x + threadIdx.y * blockDim.x] = store;
}

} // end namespace wath

