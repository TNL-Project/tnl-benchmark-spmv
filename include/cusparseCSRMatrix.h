#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/SparseMatrixBase.h>
#ifdef __CUDACC__
#include <cusparse.h>
#endif

#include <cstddef>
//#include <stdexcept>

#ifdef __CUDACC__
#define CHECK_CUSPARSE(func)                                            \
{                                                                       \
    cusparseStatus_t status = (func);                                   \
    if( status != CUSPARSE_STATUS_SUCCESS ){                            \
        std::cerr << "CUSPARSE API failed at line " << __LINE__         \
                  << " with error: " << cusparseGetErrorString(status)  \
                  << " " << status << std::endl;                        \
        throw std::runtime_error( cusparseGetErrorString(status) );     \
    }                                                                   \
}
#endif

namespace TNL {

template< typename Real >
class CusparseCSRBase
{
   public:
      using RealType = Real;
      using DeviceType = TNL::Devices::Cuda;
      using MatrixType = TNL::Matrices::SparseMatrixBase< Real,
                                                          TNL::Devices::Cuda,
                                                          int,
                                                          Matrices::GeneralMatrix,
                                                          Algorithms::Segments::CSRView< TNL::Devices::Cuda, int >,
                                                          Real >;

      CusparseCSRBase() = default;

#ifdef __CUDACC__
      void init( const MatrixType& matrix,
                 cusparseHandle_t* cusparseHandle )
      {
         this->cusparseHandle = cusparseHandle;
         this->matrix = &matrix;
#if CUDART_VERSION < 11000
         cusparseCreateMatDescr( & this->matrixDescriptor );
#endif
      };
#endif

      int getRows() const
      {
         return matrix->getRows();
      }

      int getColumns() const
      {
         return matrix->getColumns();
      }

      int getNumberOfMatrixElements() const
      {
         return matrix->getAllocatedElementsCount();
      }


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         throw std::runtime_error( "Unsupported Real type for cusparse." );
      }

   protected:

      const MatrixType* matrix = nullptr;
#ifdef __CUDACC__
      cusparseHandle_t* cusparseHandle;
#if CUDART_VERSION < 11000
      cusparseMatDescr_t matrixDescriptor;
#else
      cusparseSpMatDescr_t matA;
      TNL::Containers::Array< std::byte, TNL::Devices::Cuda > buffer;
#endif
#endif

};

template< typename Real >
class CusparseCSR
{};

template<>
class CusparseCSR< double > : public CusparseCSRBase< double >
{
   public:

#ifdef __CUDACC__
      template< typename InVector,
                typename OutVector >
      void init( MatrixType& matrix,
                 const InVector& inVector,
                 OutVector& outVector,
                 cusparseHandle_t* cusparseHandle )
      {
         CusparseCSRBase< double >::init( matrix, cusparseHandle );
#if defined __CUDACC__  && CUDART_VERSION >= 11000
         double alpha = 1.0;
         CHECK_CUSPARSE(
            cusparseCreateCsr( &this->matA,                                  // cusparseSpMatDescr_t* spMatDescr,
                               matrix.getRows(),                             // int64_t               rows,
                               matrix.getColumns(),                          // int64_t               cols,
                               matrix.getValues().getSize(),                 // int64_t               nnz,
                               matrix.getSegments().getOffsets().getData(),  // void*                 csrRowOffsets,
                               matrix.getColumnIndexes().getData(),          // void*                 csrColInd,
                               matrix.getValues().getData(),                 // void*                 csrValues,
                               CUSPARSE_INDEX_32I,                           // cusparseIndexType_t   csrRowOffsetsType,
                               CUSPARSE_INDEX_32I,                           // cusparseIndexType_t   csrColIndType,
                               CUSPARSE_INDEX_BASE_ZERO,                     // cusparseIndexBase_t   idxBase,
                               CUDA_R_64F ));                                // cudaDataType          valueType)
         CHECK_CUSPARSE(
            cusparseCreateDnVec( &this->vecX, matrix.getColumns(), (void*) inVector.getData(), CUDA_R_64F ));
         CHECK_CUSPARSE(
            cusparseCreateDnVec( &this->vecY, matrix.getRows(), (void*) outVector.getData(), CUDA_R_64F ));
         size_t buffer_size;
         CHECK_CUSPARSE(
            cusparseSpMV_bufferSize( *( this->cusparseHandle ),          // cusparseHandle_t     handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, // cusparseOperation_t  opA,
                                       &alpha,                           // const void*          alpha,
                                       this->matA,                       // cusparseSpMatDescr_t matA,
                                       this->vecX,                       // cusparseDnVecDescr_t vecX,
                                       &alpha,                           // const void*          beta,
                                       this->vecY,                       // cusparseDnVecDescr_t vecY,
                                       CUDA_R_64F,                       // cudaDataType         computeType,
                                       CUSPARSE_SPMV_ALG_DEFAULT,        // cusparseSpMVAlg_t    alg,
                                       &buffer_size ));                  // size_t*              bufferSize)
         this->buffer.setSize( buffer_size );
#else
         cusparseCreateMatDescr( & this->matrixDescriptor );
#endif

      };
#endif


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef __CUDACC__
#if CUDART_VERSION >= 11000
         double alpha = 1.0;
         CHECK_CUSPARSE(
            cusparseSpMV( *( this->cusparseHandle ),              // cusparseHandle_t     handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,       // cusparseOperation_t  opA,
                          &alpha,                                 // const void*          alpha,
                          this->matA,                             // cusparseSpMatDescr_t matA,
                          this->vecX,                             // cusparseDnVecDescr_t vecX,
                          &alpha,                                 // const void*          beta,
                          this->vecY,                             // cusparseDnVecDescr_t vecY,
                          CUDA_R_64F,                             // cudaDataType         computeType,
                          CUSPARSE_SPMV_ALG_DEFAULT,              // cusparseSpMVAlg_t    alg,
                          ( void*) this->buffer.getData() ) );    // void*                externalBuffer)
#else
	 double d = 1.0;
         double* alpha = &d;
         cusparseDcsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->getValues().getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->getValues().getData(),
                         this->matrix->getSegments().getOffsets().getData(),
                         this->matrix->getColumnIndexes().getData(),
                         inVector.getData(),
                         alpha,
                         outVector.getData() );
#endif
#endif
      }

   protected:

#if defined __CUDACC__ && CUDART_VERSION >= 11000
      cusparseSpMatDescr_t matA;
      cusparseDnVecDescr_t vecX, vecY;
#endif

};

template<>
class CusparseCSR< float > : public CusparseCSRBase< float >
{
   public:

#ifdef __CUDACC__
      template< typename InVector,
                typename OutVector >
      void init( MatrixType& matrix,
                 const InVector& inVector,
                 OutVector& outVector,
                 cusparseHandle_t* cusparseHandle )
      {
         CusparseCSRBase< float >::init( matrix, cusparseHandle );
#if defined __CUDACC__  && CUDART_VERSION >= 11000
         float alpha = 1.0;
         CHECK_CUSPARSE(
            cusparseCreateCsr( &this->matA,                                  // cusparseSpMatDescr_t* spMatDescr,
                               matrix.getRows(),                             // int64_t               rows,
                               matrix.getColumns(),                          // int64_t               cols,
                               matrix.getValues().getSize(),                 // int64_t               nnz,
                               matrix.getSegments().getOffsets().getData(),  // void*                 csrRowOffsets,
                               matrix.getColumnIndexes().getData(),          // void*                 csrColInd,
                               matrix.getValues().getData(),                 // void*                 csrValues,
                               CUSPARSE_INDEX_32I,                           // cusparseIndexType_t   csrRowOffsetsType,
                               CUSPARSE_INDEX_32I,                           // cusparseIndexType_t   csrColIndType,
                               CUSPARSE_INDEX_BASE_ZERO,                     // cusparseIndexBase_t   idxBase,
                               CUDA_R_32F ));                                // cudaDataType          valueType
         CHECK_CUSPARSE(
            cusparseCreateDnVec( &vecX, matrix.getColumns(), (void*) inVector.getData(), CUDA_R_32F ));
         CHECK_CUSPARSE(
            cusparseCreateDnVec( &vecY, matrix.getRows(), (void*) outVector.getData(), CUDA_R_32F ));
         size_t buffer_size;
         CHECK_CUSPARSE(
            cusparseSpMV_bufferSize( *( this->cusparseHandle ),        // cusparseHandle_t     handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, // cusparseOperation_t  opA,
                                     &alpha,                           // const void*          alpha,
                                     this->matA,                       // cusparseSpMatDescr_t matA,
                                     this->vecX,                       // cusparseDnVecDescr_t vecX,
                                     &alpha,                           // const void*          beta,
                                     this->vecY,                       // cusparseDnVecDescr_t vecY,
                                     CUDA_R_32F,                       // cudaDataType         computeType,
                                     CUSPARSE_SPMV_ALG_DEFAULT,        // cusparseSpMVAlg_t    alg,
                                     &buffer_size ));                  // size_t*              bufferSize
         this->buffer.setSize( buffer_size );
#else
         cusparseCreateMatDescr( & this->matrixDescriptor );
#endif

      };
#endif


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef __CUDACC__
#if CUDART_VERSION >= 11000
         float alpha = 1.0;
         CHECK_CUSPARSE(
            cusparseSpMV( *( this->cusparseHandle ),              // cusparseHandle_t     handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,       // cusparseOperation_t  opA,
                          &alpha,                                 // const void*          alpha,
                          this->matA,                             // cusparseSpMatDescr_t matA,
                          this->vecX,                             // cusparseDnVecDescr_t vecX,
                          &alpha,                                 // const void*          beta,
                          this->vecY,                             // cusparseDnVecDescr_t vecY,
                          CUDA_R_32F,                             // cudaDataType         computeType,
                          CUSPARSE_SPMV_ALG_DEFAULT,              // cusparseSpMVAlg_t    alg,
                          ( void*) this->buffer.getData() ) );    // void*                externalBuffer)
#else
         float d = 1.0;
         float* alpha = &d;
         cusparseScsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->getValues().getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->getValues().getData(),
                         this->matrix->getSegments().getOffsets().getData(),
                         this->matrix->getColumnIndexes().getData(),
                         inVector.getData(),
                         alpha,
                         outVector.getData() );
#endif
#endif
      }

   protected:
#if defined __CUDACC__ && CUDART_VERSION >= 11000
      cusparseDnVecDescr_t vecX, vecY;
      cusparseSpMatDescr_t matA;
#endif

};

} // namespace TNL
