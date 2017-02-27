#include <iostream>
#include <algorithm>

#include "resourceLayer/Matrix.hpp"

namespace watrix
{

Matrix::Matrix(int row, int col, const float* array)
{
	Matrix( row, col );
	set(array);
}

Matrix::Matrix( int row, int col):
	h_rowNum(row),
	h_colNum(col),
	size( h_rowNum * h_colNum * sizeof(float) )
{
	cudaMalloc((void**)&d_matrix, size);
	h_matrix = new float[row*col];
}

Matrix::~Matrix()
{
	cudaFree( d_matrix );
	delete[] h_matrix;
}

void Matrix::toDevice()
{
	cudaMemcpy((void*) d_matrix, (void*) h_matrix, size, cudaMemcpyHostToDevice);
}

void Matrix::toHost()
{
	cudaMemcpy((void*) h_matrix, (void*) d_matrix, size, cudaMemcpyDeviceToHost);
}	

void Matrix::print(const std::string& name)
{
	toHost();
	std::cout << name << std::endl;
	for( int i = 0; i < h_rowNum; ++i)
	{
		for( int i2 = 0; i2 < h_colNum; ++i2)
			std::cout << h_matrix[i * h_colNum + i2]  << "\t";
		std::cout << std::endl;
	}
}

void Matrix::set(const float* matrix)
{
	for( int i = 0; i < h_rowNum*h_colNum; ++i)
	{
		h_matrix[i] = matrix[i];
	}
	toDevice();
}

Matrix::Ptr Matrix::transpose(bool newMatrix)
{
	Matrix::Ptr output;
	if( newMatrix )
		output = new Matrix( h_rowNum, h_colNum );
	else
		output = this;

	dim3 blockDim( h_colNum, h_rowNum );
	wath::matrixTranspose<<<1, blockDim>>>(output->d_matrix, this->d_matrix);
		
	std::swap( output->h_colNum, output->h_rowNum );
	return output;
}

Matrix::Ptr Matrix::add(Matrix::Ptr matrix, bool newMatrix)
{
	if( (this->h_rowNum != matrix->h_rowNum) || (this->h_colNum != matrix->h_colNum) )
	{
		std::cout << "Matrix Mismatch: add" << std::endl;
		return this;
	}
	Matrix::Ptr output;
	if( newMatrix )
		output = new Matrix(this->h_rowNum, this->h_colNum);
	else
		output = this;
	
	dim3 blockDim( this->h_rowNum, this->h_colNum );
	wath::matrixMatrixAdd<<<1,blockDim>>>(output->d_matrix, this->d_matrix, matrix->d_matrix);
	return output;
}

Matrix::Ptr Matrix::multiply( float scalar, bool newMatrix)
{
	dim3 blockDim( h_rowNum, h_colNum );
	Matrix::Ptr output;
	if( newMatrix )
		output = new Matrix(h_rowNum, h_colNum);
	else
		output = this;

	wath::scalarMatrixMultiply<<<1, blockDim>>>(output->d_matrix, this->d_matrix, scalar);
	return output;
}

Matrix::Ptr Matrix::multiply( Matrix::Ptr matrix )
{
	Matrix::Ptr output = new Matrix(this->h_rowNum, matrix->h_colNum);
	if( this->h_colNum != matrix->h_rowNum )
	{
		std::cout << "Matrix Mismatch: multiply" << std::endl;
		return output;
	}
	dim3 blockDim( this->h_rowNum, matrix->h_colNum );
	wath::matrixMatrixMultiply<<<1, blockDim>>>( output->d_matrix, this->d_matrix, matrix->d_matrix, this->h_colNum);
	return output;
}

} // namespace watrix
