#include "cudaLayer/Math.cuh"

#include <string>

/* Convention
*
*/

namespace watrix
{

class Matrix
{
private:
	float* d_matrix;
	float* h_matrix;
	int h_rowNum;
	int h_colNum;
	int size;
	void toDevice();
	void toHost();
public:
	typedef Matrix* Ptr;
	Matrix( int row, int col, const float* array);
	Matrix( int row, int col );
	~Matrix();
	/*
	* if newMatrix = true then a new matrix is returned leaving the original untouched
	* otherwise, the original is modified and the original is returned.
	*/
	Matrix::Ptr transpose( bool newMatrix=true );
	Matrix::Ptr add( Matrix::Ptr matrix, bool newMatrix=true );
	Matrix::Ptr multiply( float scalar, bool newMatrix=true );
	Matrix::Ptr multiply( Matrix::Ptr matrix );

	// functions to help data management
	void print(const std::string& name="" );
	void set(const float* array);
}; // class Matrix

} // namepsace watrix
