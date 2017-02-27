#include <iostream>

#include "cudaLayer/Math.cuh"
#include "resourceLayer/Matrix.hpp"

int main(int argc, char** argv)
{
	//float* d_result;
	//float* d_leftMatrix;
	float* d_rightMatrix;
	//cudaMalloc((void**)&d_result,      2 * 3 * sizeof(float));
	//cudaMalloc((void**)&d_leftMatrix,  2 * 4 * sizeof(float));
	cudaMalloc((void**)&d_rightMatrix, 4 * 3 * sizeof(float));
	
	//float h_result[6] = {0};
	//float h_leftMatrix[8]   = { 1, 2, 3, 4, 5, 6, 7, 8 };
	float h_rightMatrix[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	//cudaMemcpy((void*) d_leftMatrix, (void*) h_leftMatrix, 8 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) d_rightMatrix, (void*) h_rightMatrix, 12 * sizeof(float), cudaMemcpyHostToDevice);
	
	//dim3 blockDim(2,3);
	//wath::matrixMatrixAdd<<< 1, blockDim >>>(d_result, d_leftMatrix, d_rightMatrix);

	dim3 blockDim(4,3);
	wath::matrixTranspose<<<1,blockDim>>>(d_rightMatrix, d_rightMatrix);
	
	//cudaMemcpy( h_result, d_result, 6 * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_rightMatrix, d_rightMatrix, 4*3 * sizeof(float), cudaMemcpyDeviceToHost );

	for( int i = 0; i < 4*3; i++)
	{
		std::cout << h_rightMatrix[i] << std::endl;
	}

	//cudaFree(d_result);
	//cudaFree(d_leftMatrix);
	cudaFree(d_rightMatrix);
/*
	watrix::Matrix::Ptr m = new watrix::Matrix(2, 3);
	float array[] = {1,2,3,4,5,6};
	m->set( array );
	
	watrix::Matrix::Ptr m1 = new watrix::Matrix(3, 2);
	m1->set( array );
	
	watrix::Matrix::Ptr n = m->multiply(m1);

	m->print( "original" );
	m1->print( "the other original" );
	n->print( "new" );

	delete( m );
	delete( n );
*/
}
